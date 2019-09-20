import collections.abc

class Instruction:
    next_id = 0
    id_type = int

    def __init__(self):
        self.id = Instruction.next_id
        Instruction.next_id = Instruction.next_id + 1
        self.op = ""
        self.dt = 'float'
        self.shape = (1,2,4,8)
        self.operands:[Instruction] = []


class Buffer:
    next_id = 0
    id_type = int

    def __init__(self, shape, dt):
        self.id = Buffer.next_id
        Buffer.next_id = Buffer.next_id + 1
        self.shape = shape
        self.dt = dt
        self.ty = 'raw'

    def convert_type(self, new_ty):
        raise("Buffer type convert raw=>{} is not supported.".format(new_ty))


class InstructionBufferAlias:
    def __init__(self):
        self.instruction_input_buffers: {Instruction.id_type: [Buffer.id_type]} = {}
        self.instruction_output_buffer: {Instruction.id_type: Buffer.id_type} = {}

    def get_output_buffer(self, instruction):
        return self.instruction_output_buffer[instruction.id]

    def analyze_instruction(self, instruction:Instruction):
        input_buffer_ids = []
        for ins_operand in instruction.operands:
            assert(isinstance(ins_operand, Instruction))
            input_operand_buffer_id = self.get_output_buffer(ins_operand)
            input_buffer_ids.append(input_operand_buffer_id)
        self.instruction_input_buffers[instruction.id] = input_buffer_ids
        self.instruction_output_buffer[instruction.id] = Buffer(instruction.shape, instruction.dt)


class InstructionBufferRelations:
    class Action:
        nop = 0
        ass = 1
        use = 2

        def __init__(self, code):
            self.code = code

    def __init__(self):
        self.relations:[Instruction, Buffer, InstructionBufferRelations.Action] = []
        self.buffer_tracer: {Buffer.id_type: [Instruction.id_type]} = {}

    def ins_ass_buf(self, ins, buf):
        self.relations.append([ins, buf, InstructionBufferRelations.Action(InstructionBufferRelations.Action.ass)])
        assert(self.buffer_tracer.get(buf.id) is None)
        self.buffer_tracer[buf.id] = [ins.id]

    def ins_use_buf(self, ins, buf):
        self.relations.append([ins, buf, InstructionBufferRelations.Action(InstructionBufferRelations.Action.use)])
        assert(self.buffer_tracer.get(buf.id) is not None)
        self.buffer_tracer[buf.id].append(ins.id)


class GlbStatus:
    location_ty = int
    size_ty = int

    def __init__(self):
        self.mem = 1024*1024 # 1M
        self.buffer_dict: {Buffer.id_type: (GlbStatus.location_ty, GlbStatus.size_ty)} = {}

    def buffer_alloc(self, buffer):
        self.buffer_dict[buffer.id] = buffer

    def buffer_dead(self, buffer):
        del self.buffer_dict[buffer.id]

    def get_buffer_size(self, buffer):
        return 0 #TODO

    def get_total_size(self):
        return sum(size for location, size in self.buffer_dict.values())






