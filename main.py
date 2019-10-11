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
    def __init__(self, instruction: Instruction):
        self.instruction = instruction

    @staticmethod
    def get_buffer_by_instruction(buffers: typing.Collection['Buffer'], instruction: Instruction) -> 'Buffer':
        for buffer in buffers:
            if id(buffer.instruction) == id(instruction):
                return buffer

    @staticmethod
    def init_buffers(instructions: typing.Collection[Instruction]):
        ret = []
        for instruction in instructions:
            ret.append(Buffer(instruction))

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
    def __init__(self, buffer: Buffer, address: int, mem_size: int, meta_data=None):
        self.buffer = buffer
        self.address: int = address
        self.mem_size: int = mem_size
        self.meta_data = meta_data

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

    def buffer_find_next_use(self, buffer: Buffer, current_idx: int):
        all_uses_of_buffer = self.livenesses[buffer].uses.keys()
        next_use = min(i for i in all_uses_of_buffer if i > current_idx)
        return next_use


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
            (
                free_mem[0], free_mem[0] + request_mem_size
                for free_mem in self.free_mem_list
                if (free_mem[1] - free_mem[0]) >= request_mem_size
            ),
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
        key=lambda ass: buffer_liveness.buffer_find_next_use(ass.buffer, current_idx),
        reverse=True
    )

    swap_list: typing.List[BufferAssignment] = []
    for assignment in assignments_ordered:
        allocator_copy.free((assignment.address, assignment.address+assignment.mem_size))
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
        if i.overlap(allocated_mem[0], allocated_mem[1]-allocated_mem[0])
    ]
    return really_needs_to_be_swap


def update_buffer_assignments(buffer_assignments: typing.Collection[BufferAssignment],
                              new_buffers: typing.Collection[Buffer]):
    for buffer_assignment in buffer_assignments:
        buffer_assignment.buffer = Buffer.get_buffer_by_instruction(new_buffers, buffer_assignment.buffer.instruction)


class GlbAssignmentState:
    def __init__(
            self, glb_size: int,
            get_instruction_output_size: typing.Callable[[Instruction], int]
    ):
        self.get_instruction_output_size = get_instruction_output_size
        self.allocator = GlbAllocator(glb_size)
        self.instructions: typing.List[Instruction] = []
        self.buffer_assignments: typing.MutableSet[BufferAssignment] = set()
        self.instruction_domain_mapping: typing.Dict[Instruction, Instruction] = {}

    def get_buffer_assignment_by_instruction(self, instruction):
        for buffer_assignment in self.buffer_assignments:
            if buffer_assignment.buffer.instruction == instruction:
                return buffer_assignment
        else:
            return None

    def get_instruction_domain_map(self):
        def instruction_domain_map(instruction: Instruction):
            return self.instruction_domain_mapping[instruction]

        return instruction_domain_map

    def add_store_buffer_instruction(self, buffer_assignment: BufferAssignment):
        instruction_which_output_this_buffer = buffer_assignment.buffer.instruction
        new_store_instr = Instruction(
            'store',
            [self.get_instruction_domain_map()(instruction_which_output_this_buffer)],
            dt=instruction_which_output_this_buffer.dt,
            shape=instruction_which_output_this_buffer.shape
        )
        self.instructions.append(new_store_instr)
        self.allocator.free((buffer_assignment.address, buffer_assignment.address+buffer_assignment.mem_size))
        self.buffer_assignments.remove(buffer_assignment)
        # remapping old_inst->new_inst to old_inst->new_store_inst
        self.instruction_domain_mapping[instruction_which_output_this_buffer] = new_store_instr

    def add_load_buffer_instruction(self, instruction):
        instruction_in_new_domain = self.get_instruction_domain_map()(instruction)
        new_load_instr = Instruction(
            'load',
            [instruction_in_new_domain],
            dt=instruction_in_new_domain.dt,
            shape=instruction_in_new_domain.shape
        )
        self.instructions.append(new_load_instr)
        alloc_result = self.allocator.alloc(self.get_instruction_output_size(new_load_instr))
        assert (alloc_result is not None)
        (address, end) = alloc_result
        mem_size = end - address
        self.buffer_assignments.add(BufferAssignment(Buffer(new_load_instr), address, mem_size))
        # remapping old_inst->new_store_inst to old_inst->new_load_inst
        self.instruction_domain_mapping[instruction] = new_load_instr

    def add_instruction(self, prototype: Instruction, prototype_buffers: typing.Collection[Buffer],
                        prototype_liveness: BufferLiveness, prototype_idx: int):
        new_instruction = prototype.copy(self.get_instruction_domain_map())
        instruction_operands_needed_loads = [
            operand
            for can_use_ddr, operand
            in zip(new_instruction.operands_can_use_ddr, new_instruction.operands)
            if not can_use_ddr and self.get_buffer_assignment_by_instruction(operand) is None
        ]
        inst_input_swap_request_buffer_size = [
            self.get_instruction_output_size(operand)
            for operand
            in instruction_operands_needed_loads
        ]
        inst_request_buffer_size = self.get_instruction_output_size(new_instruction)

        buffer_assignments_needs_to_be_swap = []
        for mem_size in [inst_request_buffer_size] + inst_input_swap_request_buffer_size:
            buffer_assignments = ensure_glb_space(
                mem_size,
                self.allocator,
                self.buffer_assignments,
                prototype_liveness,
                prototype_idx
            )
            buffer_assignments_needs_to_be_swap = buffer_assignments_needs_to_be_swap + list(buffer_assignments)

        # add store_instructions
        for buffer_assignment in buffer_assignments_needs_to_be_swap:
            self.add_store_buffer_instruction(buffer_assignment)

        # add load_instructions
        for operand in instruction_operands_needed_loads:
            self.add_load_buffer_instruction(operand)
        # update new_instruction operands pointer to load_instruction
        new_instruction = prototype.copy(self.get_instruction_domain_map())

        # add target instruction
        self.instructions.append(new_instruction)
        alloc_result = self.allocator.alloc(inst_request_buffer_size)
        assert (alloc_result is not None)
        (address, end) = alloc_result
        mem_size = end-address
        self.buffer_assignments.add(BufferAssignment(Buffer(new_instruction), address, mem_size))
        # remapping old_inst->new_store_inst to old_inst->new_load_inst
        self.instruction_domain_mapping[prototype] = new_instruction

        # remove operand_buffers where no more using
        prototype_operand_buffers = [
            Buffer.get_buffer_by_instruction(prototype_buffers, operand)
            for can_use_ddr, operand
            in zip(prototype.operands_can_use_ddr, prototype.operands)
        ]
        prototype_dead_operands = [
            prototype_buffer.instruction
            for prototype_buffer
            in prototype_operand_buffers
            if not prototype_liveness.buffer_is_alive_at_time(prototype_buffer, prototype_idx + 1)
        ]
        dead_operand_buffer_assignments = filter(
            lambda x: x is not None,
            [
                self.get_buffer_assignment_by_instruction(self.get_instruction_domain_map()(prototype_dead_operand))
                for prototype_dead_operand
                in prototype_dead_operands
            ]
        )
        for buffer_assignment in dead_operand_buffer_assignments:
            self.allocator.free((buffer_assignment.address, buffer_assignment.address+buffer_assignment.mem_size))
            self.buffer_assignments.remove(buffer_assignment)


def assignment_glb(
        instructions: typing.Sequence[Instruction],
        buffers: typing.Collection[Buffer],
        buffer_liveness: BufferLiveness,
        get_instruction_output_size: typing.Callable[[Instruction], int]
):
    glb_state = GlbAssignmentState(100 * 1024, get_instruction_output_size)
    for idx, instruction in zip(range(len(instructions)), instructions):
        glb_state.add_instruction(instruction, buffers, buffer_liveness, idx)

    return [glb_state.instructions, glb_state.buffer_assignments]


class DdrAllocator:
    def __init__(self):
        # self.buffer_assignments:typing.MutableSet[BufferAssignment] = set()
        self.free_mem_list: typing.List[typing.List[int, int]] = []
        self.max_mem_use: int = 0

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

    def alloc(self, request_mem_size: int) -> typing.Optional[typing.Tuple[int, int]]:
        result = None
        free_mem_list: typing.List[typing.List[int, int]] = []
        bottom_free_mem = None
        for free_mem in self.free_mem_list:
            if free_mem[1] == self.max_mem_use:
                bottom_free_mem = free_mem

            if request_mem_size is None:
                free_mem_list.append(free_mem)

            free_mem_size = free_mem[1] - free_mem[0]
            if request_mem_size == free_mem_size:
                request_mem_size = None
                result = free_mem
                # no rest_free_mem
            elif request_mem_size < free_mem_size:
                rest_free_mem: typing.Tuple[int, int] = GlbAllocator.alloc_from_free_mem(tuple(*free_mem),
                                                                                         request_mem_size)
                assert (rest_free_mem is not None)
                result = (free_mem[0] + request_mem_size, free_mem[1])
                request_mem_size = None
                free_mem_list.append(list(rest_free_mem))
            else:
                continue
        if request_mem_size is None:
            self.free_mem_list = free_mem_list
            return result
        else:
            if bottom_free_mem:
                lack_of_mem = request_mem_size - (bottom_free_mem[1] - bottom_free_mem[0])
                bottom_free_mem[1] = bottom_free_mem[1] + lack_of_mem
                self.max_mem_use = self.max_mem_use + lack_of_mem
                self.free_mem_list.remove(bottom_free_mem)
                return tuple(*bottom_free_mem)
            else:
                address = self.max_mem_use
                self.max_mem_use = self.max_mem_use + request_mem_size
                return address, request_mem_size


def assignment_ddr(
        instructions: typing.Sequence[Instruction],
        buffers: typing.Collection[Buffer],
        buffer_liveness: BufferLiveness,
        glb_buffer_assignments: typing.Collection[BufferAssignment],
        get_instruction_output_size: typing.Callable[[Instruction], int]
) -> typing.Collection[BufferAssignment]:
    update_buffer_assignments(glb_buffer_assignments, buffers)
    allocator = DdrAllocator()
    ddr_buffer_assignments: typing.MutableSet[BufferAssignment] = set()

    def get_buffer_assignment_by_instruction(buffer_assignments, inst):
        for buffer_assignment in buffer_assignments:
            if buffer_assignment.buffer.instruction == inst:
                return buffer_assignment
        else:
            return None

    for idx, instruction in zip(range(len(instructions)), instructions):
        # alloc instruction output memory if its not in glb
        if get_buffer_assignment_by_instruction(glb_buffer_assignments, instruction):
            pass  # this buffer is using glb not in ddr
        else:
            addr, end = allocator.alloc(get_instruction_output_size(instruction))
            mem_size = end-addr
            ddr_buffer_assignments.add(BufferAssignment(Buffer(instruction), addr, mem_size))

        # free dead operand instruction input memory
        for operand_instruction in instruction.operands:
            is_alive = buffer_liveness.buffer_is_alive_at_time(
                Buffer.get_buffer_by_instruction(buffers, operand_instruction),
                idx
            )
            if is_alive:
                continue

            if get_buffer_assignment_by_instruction(glb_buffer_assignments, operand_instruction):
                pass  # this buffer is using glb not in ddr
            else:
                ddr_buffer_assignment = get_buffer_assignment_by_instruction(
                    ddr_buffer_assignments,
                    operand_instruction
                )
                allocator.free((ddr_buffer_assignment.address, ddr_buffer_assignment.address+ddr_buffer_assignment.mem_size))

    update_buffer_assignments(ddr_buffer_assignments, buffers)
    return ddr_buffer_assignments


def run_assignment(instructions: typing.Sequence[Instruction]):
    get_instruction_output_size: typing.Callable[[Instruction], int] = lambda buffer: 1

    buffers = Buffer.init_buffers(instructions)

    buffer_liveness = BufferLiveness(instructions, buffers)

    [instructions_glb_assignment, buffer_glb_assignments] = assignment_glb(
        instructions, buffers, buffer_liveness, get_instruction_output_size
    )

    [instructions_glb_assignment_opt, buffer_glb_assignments_opt] = instructions_opt(
        instructions_glb_assignment, buffer_glb_assignments, buffer_liveness
    )
    instructions = instructions_glb_assignment_opt
    buffers = Buffer.init_buffers(instructions)
    buffer_glb_assignments = buffer_glb_assignments_opt

    buffer_liveness = BufferLiveness(instructions, buffers)
    buffer_ddr_assignments = assignment_ddr(instructions, buffers, buffer_liveness, buffer_glb_assignments,
                                            get_instruction_output_size)

    return [instructions, buffer_glb_assignments, buffer_ddr_assignments]


def main():
    run_assignment([])
