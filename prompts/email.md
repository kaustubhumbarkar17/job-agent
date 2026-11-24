You are drafting a professional email to a contact about a job opening.

<profile>
{profile_md}
</profile>

<job>
{job_json}
</job>

<contact_role>
{contact_role}
</contact_role>

Draft the four fields below. Follow these constraints:
- Total output must be under 120 words (subject + hook + proof + ask).
- No flattery, no "I hope this finds you well".
- Every number in your generated copy must appear in the profile. Use the subject field for a short proof point; the role and company are added automatically.
- Do not invent details about the company.
- The hook must tie a specific detail of this role/company to the candidate's work.
- Use concrete metrics from matched projects when the profile provides them.

Return JSON matching the schema: {subject, hook, proof, ask}.
