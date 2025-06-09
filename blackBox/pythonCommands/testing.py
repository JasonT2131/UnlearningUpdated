import re

def inputFinder(text):
    pattern = re.compile(r"- \*\*(.+?)\s*-\s*.+?:\*\*\s*(.+)")
    match = pattern.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    sentence = match.group(2).strip()
    return f"**{name}:** {sentence}"
line = '- **Jaime Vasquez - True Crime:** Jaime Vasquez specializes in the true crime genre.'


print(inputFinder(line))