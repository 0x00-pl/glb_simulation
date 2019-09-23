import math
import collections


class Instruction:
    def __init__(self, operands:collections.Sequence['Instruction']=None):
        operands = operands or []
        self.op = ""
        self.dt = 'float'
        self.shape = (1,2,4,8)
        self.operands:collections.Sequence[Instruction] = operands


class Buffer:
    class SliceInfo:
        def __init__(self, base_buffer, sliced_pos, sliced_shape):
            self.base_buffer = base_buffer
            self.sliced_pos = sliced_pos
            self.sliced_shape = sliced_shape

    def __init__(self, instruction:Instruction, buffers:collections.abc.Collection['Buffer']=None):
        self.instruction = instruction
        if self.instruction.op == 'slice':
            assert(buffers is not None)
            slice_base =  Buffer.get_buffer_by_instruction(buffers, self.instruction.operands[0])
            self.slice_info:Buffer.SliceInfo = Buffer.SliceInfo(slice_base, [0,0,0,0], [1,1,1,1])
        else:
            self.slice_info:Buffer.SliceInfo = None

    @staticmethod
    def get_buffer_by_instruction(buffers:collections.abc.Collection['Buffer'], instruction:Instruction) -> 'Buffer':
        for buffer in buffers:
            if id(buffer.instruction) == id(instruction):
                return buffer

    @staticmethod
    def init_buffers(instructions:collections.abc.Collection[Instruction]):
        ret = []
        for instruction in instructions:
            ret.append(Buffer(instruction, ret))

        return ret


class BufferAssignment:
    def __init__(self, buffer:Buffer, address:int):
        self.buffer = buffer
        self.address:int = address


def instructions_opt(instructions, buffers, buffer_assignments, buffer_liveness):
    return [instructions, buffers, buffer_assignments]


def main():
    instructions:collections.abc.Collection[Instruction] = []
    get_buffer_size:collections.Callable[[Buffer], int] = lambda buffer: 1

    buffers = Buffer.init_buffers(instructions)

    buffer_liveness = BufferLiveness(buffers, [])

    [instructions_glb_assignment, buffer_glb_assignments] = assignment_glb(instructions, buffer_liveness, get_buffer_size)
    [instructions_glb_assignment_opt, buffers_opt, buffer_glb_assignments_opt] = instructions_opt(instructions_glb_assignment, buffers, buffer_glb_assignments, buffer_liveness)
    instructions = instructions_glb_assignment_opt
    buffers = buffers_opt
    buffer_glb_assignments = buffer_glb_assignments_opt

    buffer_liveness_ddr_only = BufferLiveness(buffers, [i.buffer for i in buffer_glb_assignments])
    [instructions_ddr_assignment, buffer_ddr_assignments] = assignment_ddr(instructions, buffer_liveness_ddr_only, get_buffer_size)

    return [instructions_ddr_assignment, buffer_glb_assignments, buffer_ddr_assignments]



# class GlbStatus:
#     location_ty = int
#     size_ty = int
#
#     def __init__(self):
#         self.mem = 1024*1024 # 1M
#         self.block_size = 64
#         self.buffer_dict: {Buffer.id_type: (GlbStatus.location_ty, GlbStatus.size_ty)} = {}
#
#     @staticmethod
#     def assign_range_to_list(dst, src, pos):
#         return dst[:pos] + src + dst[pos+len(src):]
#
#     def find_space_in_mem(self, size):
#         memblock_bitmap = [False]*int(self.mem/self.block_size)
#         for location, size in self.buffer_dict.values():
#             block_loc = int(location/self.block_size)
#             block_len = int(math.ceil(size/self.block_size))
#             self.assign_range_to_list(memblock_bitmap, [True]*block_len, block_loc)
#
#         size_in_block = int(math.ceil(size/self.block_size))
#         last_free_loc = 0
#         for i in range(len(memblock_bitmap)):
#
#
#
#     def buffer_alloc(self, buffer):
#
#         self.buffer_dict[buffer.id] = buffer
#
#     def buffer_dead(self, buffer):
#         del self.buffer_dict[buffer.id]
#
#     def get_buffer_size(self, buffer):
#         return 0 #TODO
#
#     def get_total_size(self):
#         return sum(size for location, size in self.buffer_dict.values())




