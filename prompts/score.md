You are scoring how well a job fits a candidate. Use ONLY the data below.

<profile>
{profile_md}
</profile>

<job>
{job_json}
</job>

Rubric (total 100):
- must_have_skills (40): fraction of must-have skills the candidate demonstrably has.
- seniority (25): candidate has ~2.5 years. Full marks for 1-4 years required;
  half for 4-5; zero for 6+ or for intern/junior roles below their level.
- domain (20): overlap with LLM/AI products, full-stack web, backend.
- location (15): matches stated locations or remote; 0 if onsite elsewhere.

Rules:
- Only list a skill as matched if it appears in the profile.
- Do not infer skills the profile does not mention.
- Apply dealbreakers from the profile: if one applies, score = 0.
Return JSON only.
