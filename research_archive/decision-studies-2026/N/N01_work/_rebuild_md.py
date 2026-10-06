import os

p = os.path.dirname(os.path.abspath(__file__))
md_path = os.path.join(p, "t_critical_values.md")
out_path = os.path.join(p, "verify_t_critical_output.txt")

md = open(md_path, encoding="utf-8").read()
marker = "```text\n"
i = md.index(marker) + len(marker)
head = md[:i]
out = open(out_path, encoding="utf-8").read()
with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(head + out + "```\n")
print("rebuilt; head chars =", len(head), " transcript chars =", len(out))
