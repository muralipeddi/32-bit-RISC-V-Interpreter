from msilib.schema import SelfReg
import os
import argparse


MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space reason, we keep it as this large number, but the memory is still 32-bit addressable.

class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        
        with open(os.path.join(ioDir, "imem.txt")) as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress):
        # read instruction memory
        # return 32 bit hex val
        return "".join(self.IMem[ReadAddress: ReadAddress + 4])
          
class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(os.path.join(ioDir, "dmem.txt")) as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]

    def readInstr(self, ReadAddress):
        # read data memory
        # return 32 bit hex val
        return "".join(self.DMem[ReadAddress: ReadAddress + 4])
        
    def writeDataMem(self, Address, WriteData):
        # Split the 32-bit integer into four bytes and store them in little-endian order (least significant byte first)
         mask = 255
         data = []
         for i in range(4):
             data.append(WriteData & mask)
             WriteData = WriteData >> 8
         for j in range(4):
             self.DMem[Address + j] = format(data.pop(), '08b')

                     
    def outputDataMem(self):
        resPath = os.path.join(self.ioDir, f"{self.id}_DMEMResult.txt")
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])

class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = [0x0 for i in range(32)]
    
    def readRF(self, Reg_addr):
        return self.Registers[Reg_addr]
    
    def writeRF(self, Reg_addr, Wrt_reg_data):
        if Reg_addr == 0:
            return
        self.Registers[Reg_addr] = Wrt_reg_data & ((1 << 32) - 1)
         
    def outputRF(self, cycle):
        op = ["-"*70+"\n", "State of RF after executing cycle:" + str(cycle) + "\n"]
        op.extend([str(val)+"\n" for val in self.Registers])
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):
    def __init__(self):
        self.IF = {"nop": False, "PC": 0}
        self.ID = {"nop": False, "Instr": 0}
        self.EX = {"nop": False, "Read_data1": 0, "Read_data2": 0, "Imm": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "is_I_type": False, "rd_mem": 0, 
                   "wrt_mem": 0, "alu_op": 0, "wrt_enable": 0}
        self.MEM = {"nop": False, "ALUresult": 0, "Store_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "rd_mem": 0, 
                   "wrt_mem": 0, "wrt_enable": 0}
        self.WB = {"nop": False, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "wrt_enable": 0}

class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem

class SingleStageCore(Core):
    
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(os.path.join(ioDir, "SS_"), imem, dmem)
        self.opFilePath = os.path.join(ioDir, "StateResult_SS.txt")

 #Your implementation

    def step(self):
        if self.halted == True:
            return
       
        # Fetch the instruction
        instruction = self.ext_imem.readInstr(self.state.IF["PC"])

        # Decode the instruction
        self.decode(instruction)

        # Execute the instruction
        self.execute()

        # Memory Access (if needed)
        # If the instruction is a memory operation (load/store), perform memory access
        #if self.state.EX["rd_mem"] or self.state.EX["wrt_mem"]:
            #self.memoryAccess()
        next_PC = self.state.IF["PC"]
        # Write-Back            
        #self.writeBack()
        #self.halted = True
        if self.state.IF["nop"]:
            if self.state.IF["anothor_cycle"] == 0:
                self.halted = True
            self.state.IF["anothor_cycle"] = 0
            
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
            
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.state.IF["PC"] = next_PC
        self.cycle += 1
         

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

    # def determineAluOp(self, funct3, funct7=None):
         # return "add"  # Placeholder operation            

    def sign_extension(number, offset):
        if (number & (1 << offset)) != 0:
            return number - (1 << (offset + 1))
        else:
            return number

    def decode(self, instruction):
        # Assuming instruction is a 32-bit binary string
        opcode = instruction[-1:-8:-1][::-1]
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
                elif funct3 == '010':  # LW instruction
                    imm = int(instruction[0:12], 2)  # Immediate value
                    rs1 = int(instruction[12:17], 2)
                    rd = int(instruction[20:25], 2)

                    # Set control signals and operands for the LW instruction
                    self.state.EX["Rs"] = rs1
                    self.state.EX["Imm"] = imm
                    self.state.EX["Wrt_reg_addr"] = rd
                    self.state.EX["alu_op"] = 'LW'

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
                self.state.EX["Wrt_reg_addr"] = imm7
                if opcode == '0100011' and funct3 == '010': # SW instruction
                    self.state.EX["alu_op"] = 'SW'
                

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
                self.state.EX["ALUresult"] = result
                self.state.EX["jal_target"] = current_pc + imm_value
            elif alu_op == 'BEQ':
                # Perform branch if equal operation
                rs1_value = self.myRF.readRF(self.state.EX["Rs"])
                rs2_value = self.myRF.readRF(self.state.EX["Rt"])
                imm_value = self.state.EX["Imm"]
                if rs1_value == rs2_value:
                    self.state.EX["PC"] = self.state.EX["NPC"] + self.sign_extension(imm_value,12)
                else:
                    self.state.EX["PC"] = self.state.EX["NPC"] + 4

            elif alu_op == 'BNE':
                # Perform branch if not equal operation
                rs1_value = self.myRF.readRF(self.state.EX["Rs"])
                rs2_value = self.myRF.readRF(self.state.EX["Rt"])
                imm_value = self.state.EX["Imm"]
                if rs1_value != rs2_value:
                    self.state.EX["PC"] = self.state.EX["NPC"] + self.sign_extension(imm_value,12)
                else:
                    self.state.EX["PC"] = self.state.EX["NPC"] + 4


    def memoryAccess(self):
        # Retrieve the operation type determined during the execute stage
        alu_op = self.state.EX["alu_op"]

        if alu_op == 'LW':
            # Perform load operation
            address = self.state.EX["ALUresult"]  # Get the address
            loaded_data = self.ext_dmem.readDataMem(address)  # Read data from data memory
            self.state.MEM["Wrt_data"] = loaded_data  # Save the loaded data into the state

        elif alu_op == 'SW':
            # Perform store operation
            address = self.state.EX["ALUresult"]  # Get the address
            data_to_store = self.state.EX["Store_data"]  # Get the data to be stored
            self.ext_dmem.writeDataMem(address, data_to_store)  # Write the data into data memory

    def writeBack(self):
         # Check if there is a result to write back
        if self.state.WB["wrt_enable"]:
            # Get the destination register number and the data to write
            dest_reg_num = self.state.WB["Wrt_reg_addr"]
            data_to_write = self.state.WB["Wrt_data"]

            # Write the data into the destination register
            self.myRF.writeRF(dest_reg_num, data_to_write)

            # Optionally, log or update the state to indicate that write-back is complete

        
# Your implementation

        

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")
        
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)



class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(os.path.join(ioDir, "FS_"), imem, dmem)
        self.opFilePath = os.path.join(ioDir, "StateResult_FS.txt")

    def step(self):
        # Your implementation
       
        # --------------------- WB stage ---------------------
    


        # --------------------- MEM stage --------------------
        
        
        
        # --------------------- EX stage ---------------------
        
        
        
        # --------------------- ID stage ---------------------
        
        
        
        # --------------------- IF stage ---------------------
        
        self.halted = True
        if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            self.halted = True
        
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
        
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])

        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

if __name__ == "__main__":
     
    # parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    ioDir = os.path.abspath(args.iodir)
    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)
    
    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while(True):
        if not ssCore.halted:
            ssCore.step()
        
        if not fsCore.halted:
            fsCore.step()

        if ssCore.halted and fsCore.halted:
            break
    
    # dump SS and FS data mem.
    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()
