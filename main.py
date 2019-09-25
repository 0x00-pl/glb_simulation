import typing


class Instruction:
    def __init__(self, op: str, operands: typing.Sequence['Instruction'], dt: str = 'float', shape=[1, 2, 4, 6]):
        self.op = op
        self.dt = dt
        self.shape = shape
        self.operands: typing.Sequence[Instruction] = operands
        self.operands_can_use_ddr: typing.Sequence[bool] = Instruction.get_operands_can_use_ddr(self.op, self.operands)

    def copy(self, instruction_domain_mapping=lambda x: x):
        return Instruction(
            self.op,
            list(map(instruction_domain_mapping, self.operands)),
            self.dt,
            self.shape
        )

    @staticmethod
    def get_operands_can_use_ddr(op, operands):
        return [True] * len(operands)


class Buffer:
    # class SliceInfo:
    #     def __init__(self, base_buffer, sliced_pos, sliced_shape):
    #         self.base_buffer = base_buffer
    #         self.sliced_pos = sliced_pos
    #         self.sliced_shape = sliced_shape

    def __init__(self, instruction: Instruction, buffers: typing.Collection['Buffer'] = None):
        self.instruction = instruction
        # if self.instruction.op == 'slice':
        #     assert(buffers is not None)
        #     slice_base = Buffer.get_buffer_by_instruction(buffers, self.instruction.operands[0])
        #     self.slice_info: Buffer.SliceInfo = Buffer.SliceInfo(slice_base, [0, 0, 0, 0], [1, 1, 1, 1])
        # else:
        #     self.slice_info: Buffer.SliceInfo = None

    @staticmethod
    def get_buffer_by_instruction(buffers: typing.Collection['Buffer'], instruction: Instruction) -> 'Buffer':
        for buffer in buffers:
            if id(buffer.instruction) == id(instruction):
                return buffer

    @staticmethod
    def init_buffers(instructions: typing.Collection[Instruction]):
        ret = []
        for instruction in instructions:
            ret.append(Buffer(instruction, ret))

        return ret


class Computation:
    def __init__(
            self, instructions: typing.Sequence[Instruction], input_instructions: typing.Sequence[Instruction],
            output_instruction: Instruction
    ):
        self.instructions = instructions
        self.input_instructions = input_instructions
        self.output_instruction = output_instruction


class BufferAssignment:
    def __init__(self, buffer: Buffer, address: int, mem_size: int):
        self.buffer = buffer
        self.address: int = address
        self.mem_size: int = mem_size

    def overlap(self, address, mem_size):
        return self.address < address + mem_size and address < self.address + self.mem_size


class BufferLiveness:
    class Liveness:
        def __init__(self):
            self.uses: typing.Dict[int, Instruction] = {}

        def add_uses(self, idx: int, instruction: Instruction):
            self.uses[idx] = instruction

    def __init__(
            self, instructions: typing.Sequence[Instruction], buffers: typing.Collection[Buffer],
            get_buffer_size: typing.Callable[[Buffer], int] = (lambda buffer: 1)
    ):
        self.get_buffer_size = get_buffer_size
        self.timeline = instructions
        self.livenesses: typing.Dict[Buffer, BufferLiveness.Liveness] = {}
        for idx, instruction in zip(range(len(instructions)), instructions):
            buffer = Buffer.get_buffer_by_instruction(buffers, instruction)
            if buffer not in self.livenesses.keys():
                self.livenesses[buffer] = BufferLiveness.Liveness()
            else:
                self.livenesses[buffer].add_uses(idx, instruction)

            for ins in instruction.operands:
                buf = Buffer.get_buffer_by_instruction(buffers, ins)
                self.livenesses[buf].add_uses(idx, ins)

    def buffer_is_alive_at_time(self, buffer, idx):
        uses_idxs = self.livenesses[buffer].uses.keys()
        return min(uses_idxs) <= idx <= max(uses_idxs)

    def alive_buffers_at_time(self, idx):
        return [
            buffer
            for buffer, liveness
            in self.livenesses.items()
            if self.buffer_is_alive_at_time(buffer, idx)
        ]

    def total_size_of_buffers(self, buffers):
        return sum([self.get_buffer_size(i) for i in buffers])

    def max_buffer_use(self, buffer_filter=None):
        buffer_filter = buffer_filter or (lambda buffer: True)
        max_size = 0
        for idx in range(len(self.timeline)):
            buffers = [
                buffer
                for buffer
                in self.alive_buffers_at_time(idx)
                if buffer_filter(buffer)
            ]
            max_size = max(max_size, self.total_size_of_buffers(buffers))

        return max_size


def instructions_opt(instructions, buffers, buffer_assignments, buffer_liveness):
    return [instructions, buffers, buffer_assignments]


class GlbAllocator:
    def __init__(self, mem_size: int):
        self.mem_size: int = mem_size
        self.free_mem_list: typing.Collection[typing.Tuple[int, int]] = [(0, self.mem_size)]

    def copy(self):
        o = GlbAllocator(self.mem_size)
        o.free_mem_list = [(beg, end) for beg, end in self.free_mem_list]
        return o

    # return rest of free_mem
    @staticmethod
    def alloc_from_free_mem(free_mem: typing.Tuple[int, int], mem_size):
        free_mem_size = free_mem[1] - free_mem[0]
        assert (free_mem_size >= mem_size)
        if free_mem_size == mem_size:
            return None
        else:
            return free_mem[0] + mem_size, free_mem[1]

    def free(self, free_mem: typing.Tuple[int, int]):
        free_mem_list = []
        for i in self.free_mem_list:
            if i[1] == free_mem[0]:
                free_mem = (i[0], free_mem[1])
            elif i[0] == free_mem[1]:
                free_mem = (free_mem[0], i[1])
            else:
                free_mem_list.append(i)

        free_mem_list.append(free_mem)
        self.free_mem_list = free_mem_list

    def alloc_dry_run(self, request_mem_size: int) -> typing.Optional[typing.Tuple[int, int]]:
        return next(
            (free_mem for free_mem in self.free_mem_list if (free_mem[1] - free_mem[0]) >= request_mem_size),
            default=None
        )

    def alloc(self, request_mem_size: int) -> typing.Optional[typing.Tuple[int, int]]:
        result = None
        free_mem_list = []
        for free_mem in self.free_mem_list:
            if request_mem_size is None:
                free_mem_list.append(free_mem)

            free_mem_size = free_mem[1] - free_mem[0]
            if request_mem_size == free_mem_size:
                request_mem_size = None
                result = free_mem
                # no rest_free_mem
            elif request_mem_size < free_mem_size:
                rest_free_mem = GlbAllocator.alloc_from_free_mem(free_mem, request_mem_size)
                result = (free_mem[0] + request_mem_size, free_mem[1])
                request_mem_size = None
                free_mem_list.append(rest_free_mem)
            else:
                continue
        if request_mem_size is None:
            self.free_mem_list = free_mem_list
            return result
        else:
            return None


def buffer_find_next_use(buffer: Buffer, current_idx: int, buffer_liveness: BufferLiveness):
    all_uses_of_buffer = buffer_liveness.livenesses[buffer].uses.keys()
    next_use = min(i for i in all_uses_of_buffer if i > current_idx)
    return next_use


# return buffer_assignments which needs to be swap
def ensure_glb_space(
        mem_size: int,
        allocator: GlbAllocator,
        assignments: typing.Collection[BufferAssignment],
        buffer_liveness: BufferLiveness,
        current_idx: int
) -> typing.Collection[BufferAssignment]:
    if allocator.alloc_dry_run(mem_size):
        return []

    allocator_copy = allocator.copy()
    assignments_ordered = sorted(
        assignments,
        key=lambda ass: buffer_find_next_use(ass.buffer, current_idx, buffer_liveness),
        reverse=True
    )

    swap_list: typing.List[BufferAssignment] = []
    for assignment in assignments_ordered:
        allocator_copy.free((assignment.address, assignment.mem_size))
        swap_list.append(assignment)
        allocated_mem = allocator_copy.alloc_dry_run(mem_size)
        if allocated_mem is not None:
            break
    else:
        raise ValueError("not enough space in glb")

    really_needs_to_be_swap = [
        i
        for i
        in swap_list
        if i.overlap(*allocated_mem)
    ]
    return really_needs_to_be_swap


def assignment_glb_impl_request_resources(
        request_input_buffers, request_output_mem_size, latest_used_ordered_buffers
):
    return [buffers_needs_to_be_store]


def assignment_glb_impl_update_status():
    # todo
    pass


def assignment_glb_impl_add_instruction_update_buffer_assignments(instruction, buffer_assignments, new_instruction):
    # todo
    pass


def assignment_glb_new(
        instructions: typing.Sequence[Instruction],
        buffers: typing.Collection[Buffer],
        buffer_liveness: BufferLiveness,
        get_buffer_size: typing.Callable[[Buffer], int]
):
    # global_states
    # result collections

    for idx, instruction in zip(range(len(buffer_liveness.timeline)), buffer_liveness.timeline):


# get resource requirements
# request resource in global_states
# assert if resource is not enough
# insert store
# insert load
# insert copy of instruction


def assignment_glb(
        instructions: typing.Sequence[Instruction],
        buffers: typing.Collection[Buffer],
        buffer_liveness: BufferLiveness,
        get_buffer_size: typing.Callable[[Buffer], int]
):
    allocator = GlbAllocator(100 * 1024)  # glb size
    # buffers_glb: typing.Collection[Buffer] = []
    instructions_glb_assignment: typing.List[Instruction] = []
    instruction_domain_mapping: typing.Dict[Instruction, Instruction] = {}
    buffer_glb_assignments: typing.List[BufferAssignment] = []

    for idx, instruction in zip(range(len(buffer_liveness.timeline)), buffer_liveness.timeline):
        inst_request_buffer_size = get_buffer_size(Buffer.get_buffer_by_instruction(buffers, instruction))
        buffer_assignments_needs_to_be_swap = ensure_glb_space(
            inst_request_buffer_size,
            allocator,
            buffer_glb_assignments,
            buffer_liveness,
            idx
        )

        def instruction_domain_map(old_inst: Instruction):
            return instruction_domain_mapping[old_inst]

        for buf_ass in buffer_assignments_needs_to_be_swap:
            inst_of_buf_ass = buf_ass.buffer.instruction
            new_store_instr = Instruction(
                'store',
                [instruction_domain_map(inst_of_buf_ass)],
                dt=inst_of_buf_ass.dt,
                shape=inst_of_buf_ass.shape
            )
            instructions_glb_assignment.append(new_store_instr)
            # remapping old_inst->new_inst to old_inst->new_store_inst
            instruction_domain_mapping[inst_of_buf_ass] = new_store_instr
            allocator.free((buf_ass.address, buf_ass.mem_size))

        buffer_glb_assignments = [i for i in buffer_glb_assignments if i not in buffer_assignments_needs_to_be_swap]

        # add current old_inst->new_inst
        current_new_instruction = instruction.copy(instruction_domain_map)
        instruction_domain_mapping[instruction] = current_new_instruction
        instructions_glb_assignment.append(current_new_instruction)
        current_instruction_buffer = allocator.alloc(inst_request_buffer_size)

        # add operand memory
        assert (current_instruction_buffer is not None)
        buffer_glb_assignments.append(BufferAssignment(Buffer(current_new_instruction), *current_instruction_buffer))

    return [instructions_glb_assignment, buffer_glb_assignments]


def main():
    instructions: typing.Sequence[Instruction] = []
    get_buffer_size: typing.Callable[[Buffer], int] = lambda buffer: 1

    buffers = Buffer.init_buffers(instructions)

    buffer_liveness = BufferLiveness(instructions, buffers)

    [instructions_glb_assignment, buffer_glb_assignments] = assignment_glb(
        instructions, buffers, buffer_liveness, get_buffer_size
    )

    [instructions_glb_assignment_opt, buffers_opt, buffer_glb_assignments_opt] = instructions_opt(
        instructions_glb_assignment, buffers, buffer_glb_assignments, buffer_liveness
    )
    instructions = instructions_glb_assignment_opt
    buffers = buffers_opt
    buffer_glb_assignments = buffer_glb_assignments_opt

    buffer_liveness = BufferLiveness(instructions, buffers)
    buffer_ddr_assignments = assignment_ddr(instructions, buffer_liveness, buffer_glb_assignments, get_buffer_size)

    return [instructions, buffer_glb_assignments, buffer_ddr_assignments]

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
