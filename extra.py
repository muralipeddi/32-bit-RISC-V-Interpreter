def step(self):
       
        if not self.halted:
            # Fetch the instruction
            instruction = self.ext_imem.readInstr(self.state.IF["PC"])

            # Decode the instruction
            # This requires a 'decode' method that parses the instruction and updates the state
            # with relevant information like operation type, source and destination operands, etc.
            self.decode(instruction)

            # Execute the instruction
            # This requires an 'execute' method that performs the operation specified by the instruction
            # and updates the state with the result.
            self.execute()

            # Memory Access (if needed)
            # If the instruction is a memory operation (load/store), perform memory access
            if self.state.EX["rd_mem"] or self.state.EX["wrt_mem"]:
                self.memoryAccess()

            # Write-Back
            # This involves updating the register file if the instruction requires it
            self.writeBack()

            

        # Existing code to handle output and state update
        self.myRF.outputRF(self.cycle) 
        self.printState(self.nextState, self.cycle)
        self.state = self.nextState
        self.cycle += 1



def fetch(self):
        # Fetch the instruction from the instruction memory at the current PC
        current_pc = self.state.IF["PC"]
        instruction = self.ext_imem.readInstr(current_pc)

        # Check if the instruction is a HALT command
        if instruction == 'HALT':
            self.halted = True
        else:
            # Set the fetched instruction in the ID (Instruction Decode) state for the next stage
            self.state.ID["Instruction"] = instruction

            # Update the PC for the next instruction
            # In a more complex simulator, this might involve branch or jump logic
            # For a basic single-stage simulator, we'll just increment the PC by 4 (size of one instruction)
            self.state.IF["PC"] = current_pc + 4


def determineAluOp(self, funct3, funct7=None):
        # Implement logic to determine the ALU operation based on funct3 (and funct7 for R-type)
        return "add"  # Placeholder operation            

def decode(self, instruction):
        # Assuming instruction is a 32-bit binary string
        opcode = instruction[25:32]

        if opcode == '0110011':  # R-type
            funct7 = instruction[0:7]
            rs2 = int(instruction[7:12], 2)  # Source register 2
            rs1 = int(instruction[12:17], 2)  # Source register 1
            funct3 = instruction[17:20]
            rd = int(instruction[20:25], 2)  # Destination register
            # Set up control signals and operands for R-type instruction
            
            # Determine the specific R-type operation
            if funct3 == '000':
                if funct7 == '0000000':
                    alu_op = 'ADD'
                elif funct7 == '0100000':
                    alu_op = 'SUB'
            elif funct3 == '100':
                alu_op = 'XOR'
            elif funct3 == '110':
                alu_op = 'OR'
            elif funct3 == '111':
                alu_op = 'AND'
            
            self.state.EX["Rs"] = rs1
            self.state.EX["Rt"] = rs2
            self.state.EX["Wrt_reg_addr"] = rd
            self.state.EX["alu_op"] = alu_op


        elif opcode == '0010011':  # I-type
            imm = int(instruction[0:12], 2)  # Immediate value
            rs1 = int(instruction[12:17], 2)  # Source register
            funct3 = instruction[17:20]
            rd = int(instruction[20:25], 2)  # Destination register
            # Set up control signals and operands for I-type instruction
            
            # Determine the specific I-type arithmetic operation
            if funct3 == '000':
                alu_op = 'ADDI'
            elif funct3 == '100':
                alu_op = 'XORI'
            elif funct3 == '110':
                alu_op = 'ORI'
            elif funct3 == '111':
                alu_op = 'ANDI'

            # Set control signals and operands for the arithmetic I-type instruction
            self.state.EX["Rs"] = rs1
            self.state.EX["Imm"] = imm
            self.state.EX["Wrt_reg_addr"] = rd
            self.state.EX["alu_op"] = alu_op



        elif opcode == '0100011':  # S-type instructions
            imm5 = instruction[0:7]  # imm[4:0]
            imm7 = instruction[20:25]  # imm[11:5]
            imm = int(imm7 + imm5, 2)  # Combine immediate fields
            #imm = self.calculateSTypeImm(instruction)
            rs2 = int(instruction[7:12], 2)
            rs1 = int(instruction[12:17], 2)
            funct3 = instruction[17:20]
            # Set control signals and operands for S-type instructions
          
            # Set control signals and operands for the arithmetic I-type instruction
            self.state.EX["Rs"] = rs1
            self.state.EX["Rt"] = rs2
            self.state.EX["Imm"] = imm
            self.state.EX["Wrt_reg_addr"] = rd
            if opcode == '0100011' and funct3 == '010': # SW instruction
                self.state.EX["alu_op"] = 'SW'
            else:
                self.state.EX["alu_op"] = alu_op


        elif opcode == '0000011' and funct3 == '010':  # LW instruction
            imm = int(instruction[0:12], 2)  # Immediate value
            rs1 = int(instruction[12:17], 2)
            rd = int(instruction[20:25], 2)

            # Set control signals and operands for the LW instruction
            self.state.EX["Rs"] = rs1
            self.state.EX["Imm"] = imm
            self.state.EX["Wrt_reg_addr"] = rd
            self.state.EX["alu_op"] = 'LW'


        elif opcode == '1100011':  # B-type instructions
            imm11 = instruction[0]
            imm10_5 = instruction[1:7]
            imm4_1 = instruction[20:24]
            imm12 = instruction[24]
            imm = int(imm12 + imm11 + imm10_5 + imm4_1 + "0", 2)  # Combine immediate fields and extend to 32 bits
            #imm = self.calculateBTypeImm(instruction)
            rs2 = int(instruction[7:12], 2)
            rs1 = int(instruction[12:17], 2)
            funct3 = instruction[17:20]
            # Set control signals and operands for B-type instructions
            
            if funct3 == '000':  # BEQ
                alu_op = 'BEQ'
            elif funct3 == '001':  # BNE
                alu_op = 'BNE'
            # Set control signals and operands for BEQ/BNE
            self.state.EX["Rs"] = rs1
            self.state.EX["Rt"] = rs2
            self.state.EX["Imm"] = imm
            self.state.EX["alu_op"] = alu_op

       
        elif opcode == '0110111' or opcode == '0010111':  # U-type instructions
            imm = int(instruction[0:20], 2) << 12  # Immediate value, shifted left by 12 bits
            rd = int(instruction[20:25], 2)  # Destination register
            # Set control signals and operands for U-type instructions

        
        elif opcode == '1101111':  # J-type instructions
            imm20 = instruction[0]
            imm10_1 = instruction[1:11]
            imm11 = instruction[11]
            imm19_12 = instruction[12:20]
            imm = int(imm20 + imm19_12 + imm11 + imm10_1 + "0", 2)  # Combine immediate fields and extend to 32 bits
            rd = int(instruction[20:25], 2)  # Destination register
            # Set control signals and operands for J-type instructions
            self.state.EX["Imm"] = imm
            self.state.EX["Wrt_reg_addr"] = rd
            self.state.EX["alu_op"] = 'JAL'
        
        
        elif instruction == 'HALT':  # Special case for HALT
            # Handle HALT execution
            self.halted = True
 


def execute(self):
        # Get the operation type
        alu_op = self.state.EX["alu_op"]

        if alu_op == 'ADD':
            # Perform addition operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            rs2_value = self.myRF.readRF(self.state.EX["Rt"])
            result = rs1_value + rs2_value
            self.state.EX["result"] = result

        elif alu_op == 'SUB':
            # Perform subtraction operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            rs2_value = self.myRF.readRF(self.state.EX["Rt"])
            result = rs1_value - rs2_value
            self.state.EX["result"] = result

        elif alu_op == 'XOR':
            # Perform bitwise XOR operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            rs2_value = self.myRF.readRF(self.state.EX["Rt"])
            result = rs1_value ^ rs2_value
            self.state.EX["result"] = result

        elif alu_op == 'OR':
            # Perform bitwise OR operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            rs2_value = self.myRF.readRF(self.state.EX["Rt"])
            result = rs1_value | rs2_value
            self.state.EX["result"] = result

        elif alu_op == 'AND':
            # Perform bitwise AND operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            rs2_value = self.myRF.readRF(self.state.EX["Rt"])
            result = rs1_value & rs2_value
            self.state.EX["result"] = result

        elif alu_op == 'ADDI':
            # Perform addition immediate operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            imm_value = self.state.EX["Imm"]
            result = rs1_value + imm_value
            self.state.EX["result"] = result

        elif alu_op == 'XORI':
            # Perform bitwise XOR immediate operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            imm_value = self.state.EX["Imm"]
            result = rs1_value ^ imm_value
            self.state.EX["result"] = result

        elif alu_op == 'ORI':
            # Perform bitwise OR immediate operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            imm_value = self.state.EX["Imm"]
            result = rs1_value | imm_value
            self.state.EX["result"] = result

        elif alu_op == 'ANDI':
            # Perform bitwise AND immediate operation
            rs1_value = self.myRF.readRF(self.state.EX["Rs"])
            imm_value = self.state.EX["Imm"]
            result = rs1_value & imm_value
            self.state.EX["result"] = result

        elif alu_op == 'JAL':
            # Perform jump and link operation
            current_pc = self.state.ID["NPC"]
            imm_value = self.state.EX["Imm"]
            result = current_pc + 4
            self.state.EX["result"] = result
            self.state.EX["jal_target"] = current_pc + imm_value



def memoryAccess(self):
    # Retrieve the operation type determined during the execute stage
    alu_op = self.state.EX["alu_op"]

    if alu_op == 'LW':
        # Perform load operation
        address = self.state.EX["ALUresult"]  # Get the address
        loaded_data = self.myDataMem.readInstr(address)  # Read data from data memory
        self.state.MEM["Wrt_data"] = loaded_data  # Save the loaded data into the state

    elif alu_op == 'SW':
        # Perform store operation
        address = self.state.EX["ALUresult"]  # Get the address
        data_to_store = self.state.EX["Store_data"]  # Get the data to be stored
        self.myDataMem.writeDataMem(address, data_to_store)  # Write the data into data memory



def writeBack(self):
    # Check if there is a result to write back
    if self.state.WB["wrt_enable"]:
        # Get the destination register number and the data to write
        dest_reg_num = self.state.WB["Wrt_reg_addr"]
        data_to_write = self.state.WB["Wrt_data"]

        # Write the data into the destination register
        self.myRF.writeRF(dest_reg_num, data_to_write)

        # Optionally, log or update the state to indicate that write-back is complete








    
      
        
       
            
