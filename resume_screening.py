# Required skills
skills = ["python", "machine learning", "sql", "ai"]

print("Resume Screening AI")

# User resume input
resume = input("Enter resume text: ").lower()

matched_skills = []

# Check skills
for skill in skills:
    if skill in resume:
        matched_skills.append(skill)

# Output
print("\nMatched Skills:")
for skill in matched_skills:
    print("-", skill)

# Result
if len(matched_skills) >= 2:
    print("\nCandidate Shortlisted")
else:
    print("\nCandidate Not Shortlisted")
