SYS_INST = "You are a security expert that is good at static program analysis."

SYS_INST_COT = SYS_INST

PROMPT_INST = """Please analyze the following code:
```
{func}
```
Please indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability. 

Only reply with one of the options above. Do not include any further information.
"""

PROMPT_INST_COT = PROMPT_INST
