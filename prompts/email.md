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
- Every number in your output must appear in the profile or job data.
- Do not invent details about the company.
- The hook must tie a specific detail of this role/company to the candidate's work.
- The proof must include two short sentences with concrete metrics from matched projects.

Return JSON matching the schema: {subject, hook, proof, ask}.
