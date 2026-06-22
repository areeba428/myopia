"""Create a Word-targeted copy of the report tex with the TikZ figure replaced
by the rendered PNG (Word cannot render TikZ)."""
import shutil

shutil.copyfile("My_Study_Report.tex", "My_Study_Report_word.tex")
t = open("My_Study_Report_word.tex", encoding="utf-8").read()
start = t.index(r"\resizebox")
end_marker = r"\input{pipeline_body}"
ei = t.index(end_marker) + len(end_marker)
j = ei
while t[j] in " \r\n\t":
    j += 1
assert t[j] == "}", repr(t[j])
j += 1
replacement = r"\includegraphics[width=\textwidth]{../diagram/pipeline-1.png}"
open("My_Study_Report_word.tex", "w", encoding="utf-8").write(t[:start] + replacement + t[j:])
print("word tex ready:", "pipeline-1.png" in open("My_Study_Report_word.tex", encoding="utf-8").read())
