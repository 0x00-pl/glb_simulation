import typing


class Instruction:
    def __init__(self, op: str, operands: typing.Sequence['Instruction'], shape=[1, 2, 4, 6]):
        self.op = op
        self.shape = shape
        self.operands: typing.List[Instruction] = list(operands)
        # self.operands_can_use_ddr: typing.List[bool] = [False]*len(operands)
        # self.output_can_use_ddr: bool = False
        self.operand_output_idx: typing.Optional[int] = None
        self.is_virtual = False


def instruction_output_to_mem_size(instruction: Instruction):
    return 100


class Span:
    def __init__(self, start: int, length: int):
        self.start = start
        self.length = length


class BufferUse:
    def __init__(self, used_by_instruction: Instruction, used_by_operand_idx: int,
                 use_for_output: bool = False):
        self.used_by_instruction = used_by_instruction
        self.used_by_operand_idx = used_by_operand_idx
        self.use_for_output = use_for_output
        self.span: typing.Optional[Span] = None


class Buffer:
    def __init__(self, instruction: Instruction):
        self.instruction = instruction
        self.uses: typing.MutableSet[BufferUse] = set()
        self.is_sub_buffer_of: Buffer = None
        self.pin_in_ddr = False
        self.buffer_type = 'ddr'
        self.ddr_span: typing.Optional[Span] = None
        self.add_uses(BufferUse(instruction, None, use_for_output=True))

    def add_uses(self, buffer_use: BufferUse):
        self.uses.add(buffer_use)


def get_buffer_by_buffer_use(buffer_set: typing.MutableSet[Buffer], buffer_use: BufferUse):
    for buffer in buffer_set:
        if buffer_use in buffer.uses:
            return buffer
    raise ValueError()


def get_buffer_by_instruction_output(buffer_set: typing.MutableSet[Buffer], instruction: Instruction):
    for buffer in buffer_set:
        for bu in buffer.uses:
            if bu.use_for_output and bu.used_by_instruction == instruction:
                return buffer
    return None


class InstructionChunk:
    def __init__(self):
        self.instructions: typing.List[Instruction] = []
        self.buffer_set: typing.MutableSet[Buffer] = set()
        self.mem_used: int = 0

    def alloc(self, mem_size):
        ret = Span(self.mem_used, mem_size)
        self.mem_used = self.mem_used + mem_size
        return ret

    def add_instruction(self, instruction: Instruction, mem_capacity: int,
                        buffer_set: typing.MutableSet[Buffer]) -> bool:
        mem_request = 0
        if not instruction.is_virtual:
            mem_request_list = [
                instruction_output_to_mem_size(i)
                for i
                in instruction.operands + ([instruction] if instruction.operand_output_idx else [])
            ]
            mem_request = sum(mem_request_list)
            assert (mem_request < mem_capacity)

        if self.mem_used + mem_request > mem_capacity:
            return False
        else:
            self.instructions.append(instruction)

            if instruction.operand_output_idx is None:
                # output new buffer
                output_buffer = Buffer(instruction)
                buffer_set.add(output_buffer)
            else:
                output_instruction = instruction.operands[instruction.operand_output_idx]
                output_buffer = get_buffer_by_instruction_output(
                    buffer_set,
                    output_instruction
                )
                output_buffer.add_uses(BufferUse(instruction, instruction.operand_output_idx, True))

            for idx, operand in enumerate(instruction.operands):
                # assign span, add uses
                buffer = get_buffer_by_instruction_output(buffer_set, operand)
                buffer.add_uses(BufferUse(instruction, idx))
                if instruction.op == 'aggreation':
                    buffer.is_sub_buffer_of = output_buffer

            if instruction.op == 'slice':
                output_buffer.is_sub_buffer_of = get_buffer_by_instruction_output(buffer_set, instruction.operands[0])

            if instruction.op in ('const', 'input', 'output'):
                output_buffer.pin_in_ddr = True

            self.buffer_set.add(output_buffer)

            return True


def get_buffer_use_by_instruction(buffer_set: typing.MutableSet[Buffer], instruction: Instruction, idx: int):
    for buffer in buffer_set:
        for buffer_use in buffer.uses:
            if buffer_use.used_by_operand_idx == idx and buffer_use.used_by_instruction == instruction:
                return buffer_use
    raise ValueError()


def build_instruction_chunks(instruction_list: typing.Sequence[Instruction], mem_capacity: int):
    chunk_list: typing.List[InstructionChunk] = []
    buffer_set: typing.Set[Buffer] = set()

    current_chunk = InstructionChunk()
    for i in range(len(instruction_list)):
        add_instruction_success = current_chunk.add_instruction(instruction_list[i], mem_capacity, buffer_set)
        if not add_instruction_success:
            chunk_list.append(current_chunk)
            current_chunk = InstructionChunk()
            current_chunk.add_instruction(instruction_list[i], mem_capacity, buffer_set)

    chunk_list.append(current_chunk)
    return chunk_list, buffer_set


def get_chunk_by_instruction(chunk_list: typing.List[InstructionChunk], instruction: Instruction):
    for chunk in chunk_list:
        if instruction in chunk.instructions:
            return chunk


def opt_buffer_type(buffer_set: typing.MutableSet[Buffer], chunk_list: typing.List[InstructionChunk]):
    for buffer in buffer_set:
        if buffer.is_sub_buffer_of or buffer.pin_in_ddr:
            continue

        chunk_set_using_buffer = set()
        for ch in chunk_list:
            if buffer in ch.buffer_set:
                chunk_set_using_buffer.add(ch)

        if len(chunk_set_using_buffer) == 1:
            buffer.buffer_type = 'local'

    for buffer in buffer_set:
        bp = buffer.is_sub_buffer_of
        if bp is not None and bp.buffer_type == 'local' and buffer.pin_in_ddr is False:
            buffer.buffer_type = 'local'


def assign_local_buffer(buffer_set: typing.MutableSet[Buffer], chunk_list: typing.List[InstructionChunk]):
    # reset_chunk mem list
    for chunk in chunk_list:
        chunk.mem_used = 0
    # find all buffer is local
    for buffer in buffer_set:
        if buffer.buffer_type != 'local':
            continue
        # find free space in chunk
        chunk = get_chunk_by_instruction(chunk_list, buffer.instruction)
        mem_size = instruction_output_to_mem_size(buffer.instruction)
        span = Span(chunk.mem_used, mem_size)
        chunk.mem_used = chunk.mem_used + mem_size

        # assign for all reference
        for bu in buffer.uses:
            bu.span = span


def link_buffer_use_in_same_chunk_is_same(buffer_set: typing.MutableSet[Buffer],
                                          chunk_list: typing.List[InstructionChunk]):
    for chunk in chunk_list:
        buffer_bu: typing.MutableMapping[Buffer, typing.Set[BufferUse]] = {}
        for buffer in chunk.buffer_set:
            for bu in buffer.uses:
                # get all uses for non-virtual instruction
                if bu.used_by_instruction in chunk.instructions and bu.used_by_instruction.is_virtual is False:
                    if buffer_bu.get(buffer, None) is None:
                        buffer_bu[buffer] = set()
                    buffer_bu[buffer].add(bu)

        for buffer, bu_set in buffer_bu.items():
            span = None
            for bu in bu_set:
                if bu.span is not None:
                    span = bu.span
            if span is None:
                span = chunk.alloc(instruction_output_to_mem_size(buffer.instruction))
            for bu in bu_set:
                bu.span = span


def assign_local_buffer_for_non_virtual_instruction(buffer_set: typing.MutableSet[Buffer],
                                                    chunk_list: typing.List[InstructionChunk]):
    for chunk in chunk_list:
        for instruction in chunk.instructions:
            # find all non-virtual instruction
            if instruction.is_virtual:
                continue
            # find span which is not assigned
            for buffer in chunk.buffer_set:
                for buffer_use in buffer.uses:
                    if buffer_use.span is None:
                        operand_output_idx = buffer_use.used_by_instruction.operand_output_idx
                        if buffer_use.use_for_output and operand_output_idx is not None:
                            # outputs use same buffer of operand
                            buffer_use.span = get_buffer_use_by_instruction(buffer_set, buffer_use.used_by_instruction,
                                                                            operand_output_idx).span
                        else:
                            mem_size = instruction_output_to_mem_size(buffer.instruction)
                            span = Span(chunk.mem_used, mem_size)
                            chunk.mem_used = chunk.mem_used + mem_size
                            # assign
                            buffer_use.span = span


def assign_ddr_buffer_which_is_not_sub_buffer(buffer_set: typing.MutableSet[Buffer],
                                              chunk_list: typing.List[InstructionChunk]):
    ddr_mem = 0
    # find all non-sub-buffer
    for buffer in buffer_set:
        if buffer.buffer_type == 'ddr':
            # assign it in ddr
            mem_size = instruction_output_to_mem_size(buffer.instruction)
            buffer.ddr_span = Span(ddr_mem, mem_size)
            ddr_mem = ddr_mem + mem_size


def link_instruction_post(src: Instruction, new_instruction: Instruction, instructions):
    new_instruction.operands = [src]
    for instruction in instructions:
        for idx, operand in enumerate(instruction):
            if operand == src:
                instruction.operands[idx] = new_instruction


def gen_code_from_chunk_list(buffer_set: typing.MutableSet[Buffer], chunk_list: typing.List[InstructionChunk]):
    for chunk in chunk_list:
        # collect all loads and stores
        buffer_loads = set()
        for instruction in chunk.instructions:
            if instruction.is_virtual is False:
                for operand in instruction.operands:
                    buffer = get_buffer_by_instruction_output(chunk.buffer_set, operand)
                    buffer_loads.add(buffer)
        for buffer in buffer_loads:
            link_instruction_post(
                buffer.instruction,
                Instruction('load', [], buffer.instruction.shape),
                chunk.instructions
            )
# gen all computes and virtual-instructions
