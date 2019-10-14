'''

instruction.is_virtual

buffer_ddr.is_sub_buffer
buffer_ddr.uses



instruction inplace

for instruction

 check glb space request for operand
 check glb space request for output
 find buffers needs to store,
 +store cold buffers
 +load operand buffers
 +run instruction
 +clean(join) dead buffers
 add dead buffers to dead_buffer_list


for opt
 +move store up
 +move clean(join) down

'''