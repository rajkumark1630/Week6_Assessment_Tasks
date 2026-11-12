
import os
from docx import Document

os.makedirs("documents", exist_ok=True)

# 1. Employee Policy - TXT
with open("documents/employee_policy.txt", "w", encoding="utf-8") as f:
    f.write("""
EMPLOYEE POLICY

Employee Health Insurance:
All employees receive health insurance benefits.
The company provides medical coverage for eligible employees.
Employees can contact HR for insurance information.

Employee Working Hours:
Normal working hours are from 9 AM to 6 PM.
Employees must follow company attendance policies.

Employee Responsibilities:
Employees must maintain professional behavior.
All employees must follow company rules.
""")

# 2. Safety Manual - TXT
with open("documents/safety_manual.txt", "w", encoding="utf-8") as f:
    f.write("""
WORKPLACE SAFETY MANUAL

Fire Safety:
Employees must know emergency exit locations.
Fire extinguishers must be accessible.
Never block emergency exits.

First Aid:
First aid kits are available in every department.
Report workplace injuries immediately.

Emergency Procedures:
Employees must follow evacuation instructions.
Contact emergency services when necessary.
""")

# 3. Security Manual - TXT
with open("documents/security_manual.txt", "w", encoding="utf-8") as f:
    f.write("""
COMPANY SECURITY MANUAL

Password Security:
Employees must use strong passwords.
Passwords must never be shared.
Enable multi-factor authentication.

Data Protection:
Company information must remain confidential.
Sensitive files must be protected.
Unauthorized access is prohibited.
""")

# 4. Leave Policy - DOCX
doc = Document()

doc.add_heading("LEAVE POLICY", 0)

doc.add_heading("Annual Leave", level=1)
doc.add_paragraph(
    "Employees are eligible for annual leave according "
    "to company policy. Leave requests must be approved "
    "by the manager."
)

doc.add_heading("Sick Leave", level=1)
doc.add_paragraph(
    "Employees can request sick leave when they are unwell. "
    "Medical documentation may be required."
)

doc.add_heading("Public Holidays", level=1)
doc.add_paragraph(
    "The company observes officially declared public holidays."
)

doc.save("documents/leave_policy.docx")

# 5. Company Rules - TXT
with open("documents/company_rules.txt", "w", encoding="utf-8") as f:
    f.write("""
COMPANY RULES

Code of Conduct:
Employees must behave respectfully.
Harassment and discrimination are prohibited.

Attendance:
Employees must arrive on time.
Attendance records must be accurate.

Confidentiality:
Employees must protect company information.
Company documents must not be shared externally.
""")

print("5 sample documents created successfully!")

print("\nDocuments folder:")

for filename in os.listdir("documents"):
    print("-", filename)
