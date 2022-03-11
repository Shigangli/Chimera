# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .start_stage import StartingStage
from .intermediate_stage import IntermediateStage
from .end_stage import EndingStage

def arch():
    return "gpt2"

def model(config, criterion):
    return [
        (lambda: StartingStage(config), ["input_ids"], ["out0"]),
        (lambda: IntermediateStage(config), ["out0"], ["out1"]),
        (lambda: IntermediateStage(config), ["out1"], ["out2"]),
        (lambda: IntermediateStage(config), ["out2"], ["out3"]),
        (lambda: IntermediateStage(config), ["out3"], ["out4"]),
        (lambda: IntermediateStage(config), ["out4"], ["out5"]),
        (lambda: IntermediateStage(config), ["out5"], ["out6"]),
        (lambda: IntermediateStage(config), ["out6"], ["out7"]),
        (lambda: IntermediateStage(config), ["out7"], ["out8"]),
        (lambda: IntermediateStage(config), ["out8"], ["out9"]),
        (lambda: IntermediateStage(config), ["out9"], ["out10"]),
        (lambda: IntermediateStage(config), ["out10"], ["out11"]),
        (lambda: IntermediateStage(config), ["out11"], ["out12"]),
        (lambda: IntermediateStage(config), ["out12"], ["out13"]),
        (lambda: IntermediateStage(config), ["out13"], ["out14"]),
        (lambda: IntermediateStage(config), ["out14"], ["out15"]),
        (lambda: IntermediateStage(config), ["out15"], ["out16"]),
        (lambda: IntermediateStage(config), ["out16"], ["out17"]),
        (lambda: IntermediateStage(config), ["out17"], ["out18"]),
        (lambda: IntermediateStage(config), ["out18"], ["out19"]),
        (lambda: IntermediateStage(config), ["out19"], ["out20"]),
        (lambda: IntermediateStage(config), ["out20"], ["out21"]),
        (lambda: IntermediateStage(config), ["out21"], ["out22"]),
        (lambda: IntermediateStage(config), ["out22"], ["out23"]),
        (lambda: IntermediateStage(config), ["out23"], ["out24"]),
        (lambda: IntermediateStage(config), ["out24"], ["out25"]),
        (lambda: IntermediateStage(config), ["out25"], ["out26"]),
        (lambda: IntermediateStage(config), ["out26"], ["out27"]),
        (lambda: IntermediateStage(config), ["out27"], ["out28"]),
        (lambda: IntermediateStage(config), ["out28"], ["out29"]),
        (lambda: IntermediateStage(config), ["out29"], ["out30"]),
        (lambda: IntermediateStage(config), ["out30"], ["out31"]),
        (lambda: IntermediateStage(config), ["out31"], ["out32"]),
        (lambda: IntermediateStage(config), ["out32"], ["out33"]),
        (lambda: IntermediateStage(config), ["out33"], ["out34"]),
        (lambda: IntermediateStage(config), ["out34"], ["out35"]),
        (lambda: IntermediateStage(config), ["out35"], ["out36"]),
        (lambda: IntermediateStage(config), ["out36"], ["out37"]),
        (lambda: IntermediateStage(config), ["out37"], ["out38"]),
        (lambda: IntermediateStage(config), ["out38"], ["out39"]),
        (lambda: IntermediateStage(config), ["out39"], ["out40"]),
        (lambda: IntermediateStage(config), ["out40"], ["out41"]),
        (lambda: IntermediateStage(config), ["out41"], ["out42"]),
        (lambda: IntermediateStage(config), ["out42"], ["out43"]),
        (lambda: IntermediateStage(config), ["out43"], ["out44"]),
        (lambda: IntermediateStage(config), ["out44"], ["out45"]),
        (lambda: IntermediateStage(config), ["out45"], ["out46"]),
        (lambda: IntermediateStage(config), ["out46"], ["out47"]),
        (lambda: IntermediateStage(config), ["out47"], ["out48"]),
        (lambda: IntermediateStage(config), ["out48"], ["out49"]),
        (lambda: IntermediateStage(config), ["out49"], ["out50"]),
        (lambda: IntermediateStage(config), ["out50"], ["out51"]),
        (lambda: IntermediateStage(config), ["out51"], ["out52"]),
        (lambda: IntermediateStage(config), ["out52"], ["out53"]),
        (lambda: IntermediateStage(config), ["out53"], ["out54"]),
        (lambda: IntermediateStage(config), ["out54"], ["out55"]),
        (lambda: IntermediateStage(config), ["out55"], ["out56"]),
        (lambda: IntermediateStage(config), ["out56"], ["out57"]),
        (lambda: IntermediateStage(config), ["out57"], ["out58"]),
        (lambda: IntermediateStage(config), ["out58"], ["out59"]),
        (lambda: IntermediateStage(config), ["out59"], ["out60"]),
        (lambda: IntermediateStage(config), ["out60"], ["out61"]),
        (lambda: IntermediateStage(config), ["out61"], ["out62"]),
        (lambda: EndingStage(config), ["out62"], ["out63"]),
        (lambda: criterion, ["out63"], ["loss"])
    ]