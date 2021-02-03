from typing import Any
import torch
import cupy
import re


@cupy.memoize(for_each_device=True)
def cupy_launch(function, kernel):
    return cupy.cuda.compile_with_cache(kernel).get_function(function)


kernel_count_classes_per_instance = '''
    extern "C" __global__ void kernel_count_classes_per_instance(
        const int n, 
        const long* sem_seg, 
        const long* ins_seg, 
        const int* is_thing_arr, 
        int* ins_seg_count_mat, 
        int* stuff_areas_count, 
        const int h, 
        const int w, 
        const int num_classes
    ) { 
        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int i = ( intIndex / w) % h;
            const int j = intIndex % w;
            
            const int ins_id = ins_seg[i * w + j];
            const int sem_class = sem_seg[i * w + j];
 
            if(ins_id > 0 && is_thing_arr[sem_class] == 1) {
                atomicAdd(&ins_seg_count_mat[ins_id * num_classes + sem_class], 1);
            }
 
            atomicAdd(&stuff_areas_count[sem_class], 1);
        }
    }
'''

kernel_paste_instance_and_semantic = '''
    extern "C" __global__ void kernel_paste_instance_and_semantic(
        const int n, 
        const long* sem_seg, 
        const long* ins_seg, 
        long* pan_seg, 
        const long* instance_classes, 
        const int* semseg_areas,
        const int* thing_list, 
        const int label_divisor, 
        const int stuff_area,
        const int h,
        const int w
    ){
        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int i = ( intIndex / w) % h;
            const int j = intIndex % w;
            const int ins_id = ins_seg[i * w + j];
            const int sem_class = sem_seg[i * w + j];
            
            if(ins_id > 0 && thing_list[sem_class] == 1) {
                pan_seg[i * w + j] = instance_classes[ins_id] * label_divisor + ins_id;
            }
            else if(ins_id == 0 && thing_list[sem_class] == 0 && semseg_areas[sem_class] >= stuff_area) {
                pan_seg[i * w + j] = sem_class * label_divisor;
            }
        }
    }
'''


class _FunctionCountInstanceClassesAndStuffArea(torch.autograd.Function):
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def forward(self, sem_seg, ins_seg, is_thing_arr, max_instances, num_classes):
        sem_seg = sem_seg.squeeze()
        ins_seg = ins_seg.squeeze()
        ss_h, ss_w = sem_seg.shape
        ins_h, ins_w = ins_seg.shape
        assert (ss_h == ins_h)
        assert (ss_w == ins_w)

        ins_seg_count_mat = torch.zeros([max_instances, num_classes], dtype=torch.int32).cuda()
        stuff_areas_count_mat = torch.zeros(num_classes, dtype=torch.int32).cuda()

        if sem_seg.is_cuda:
            n = sem_seg.nelement()
            cupy_launch('kernel_count_classes_per_instance', kernel_count_classes_per_instance)(
                grid=tuple([int((n + 512 - 1) / 512), ]),
                block=tuple([512, ]),
                args=[n, sem_seg.data_ptr(), ins_seg.data_ptr(), is_thing_arr.data_ptr(), ins_seg_count_mat.data_ptr(),
                      stuff_areas_count_mat.data_ptr(), ss_h, ss_w, num_classes]
            )
        else:
            raise NotImplementedError()
        return ins_seg_count_mat, stuff_areas_count_mat


class _FunctionPasteInstanceAndSemantic(torch.autograd.Function):
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def forward(self, sem_seg, ins_seg, pan_seg, ins_classes,
                semseg_areas, is_thing_arr, stuff_area, void_label, label_divisor):
        sem_seg = sem_seg.squeeze()
        ins_seg = ins_seg.squeeze()
        pan_seg = pan_seg.squeeze()
        ss_h, ss_w = sem_seg.shape
        ins_h, ins_w = ins_seg.shape
        pan_h, pan_w = pan_seg.shape
        assert (pan_h == ss_h == ins_h)
        assert (pan_w == ss_w == ins_w)

        if sem_seg.is_cuda:
            n = sem_seg.nelement()
            cupy_launch('kernel_paste_instance_and_semantic', kernel_paste_instance_and_semantic)(
                grid=tuple([int((n + 512 - 1) / 512)]),
                block=tuple([512]),
                args=[n, sem_seg.data_ptr(), ins_seg.data_ptr(), pan_seg.data_ptr(), ins_classes.data_ptr(),
                      semseg_areas.data_ptr(), is_thing_arr.data_ptr(), label_divisor, stuff_area, ss_h, ss_w]
            )
        else:
            raise NotImplementedError()
        return pan_seg


def count_classes_per_instance_and_stuff_areas(sem_seg, ins_seg, is_thing_arr, max_instances, num_classes):
    return _FunctionCountInstanceClassesAndStuffArea.apply(sem_seg, ins_seg, is_thing_arr, max_instances, num_classes)


def merge_instance_and_semantic(sem_seg, ins_seg, pan_seg, ins_classes, semseg_areas, is_thing_arr, stuff_area,
                                void_label, label_divisor):
    return _FunctionPasteInstanceAndSemantic.apply(sem_seg, ins_seg, pan_seg, ins_classes, semseg_areas,
                                                   is_thing_arr, stuff_area, void_label, label_divisor)
