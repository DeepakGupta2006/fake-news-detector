"""
Generates a small SYNTHETIC demo dataset (sample_data.csv) so the whole
pipeline can be run end-to-end without needing to download anything.

This is NOT a real dataset and models trained on it will NOT be reliable
for real-world use. It exists purely so you can test that every script
in this project runs correctly before you plug in a real dataset.

To get real accuracy numbers, download the Kaggle "Fake and Real News"
dataset (Fake.csv + Real.csv) from:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
and replace data/sample_data.csv with a combined file that has the same
two columns: "text" and "label" (label = "REAL" or "FAKE").
"""

import csv
import random

random.seed(42)

real_subjects = [
    "The Federal Reserve", "The city council", "Researchers at the university",
    "The Ministry of Health", "Local officials", "The company's board",
    "Election officials", "The World Health Organization", "State legislators",
    "The central bank", "The department of transportation", "Scientists",
]

real_actions = [
    "announced a quarterly report on economic growth",
    "approved a new budget for infrastructure repairs",
    "published a peer-reviewed study on vaccine efficacy",
    "released updated guidelines for public health measures",
    "confirmed the results of the recent audit",
    "held a press conference to discuss policy changes",
    "voted to extend funding for public schools",
    "issued a statement clarifying the new regulations",
    "presented data on employment trends for the past quarter",
    "opened a public comment period on the proposed changes",
]

fake_subjects = [
    "Anonymous insiders", "A viral post", "Secret documents", "Unnamed sources",
    "A shocking leak", "Conspiracy theorists", "An unverified blog",
    "A mysterious whistleblower", "Internet rumors", "A fringe website",
]

fake_actions = [
    "claim the government is hiding a miracle cure",
    "reveal that the moon landing was staged all along",
    "warn that a secret elite group controls the weather",
    "allege that vaccines contain mind-control microchips",
    "insist that aliens have already taken over world governments",
    "expose a hidden plot to replace world leaders with robots",
    "suggest that the election was secretly rigged by lizard people",
    "claim doctors are hiding a cure for aging you won't believe",
    "reveal shocking secret that banks don't want you to know",
    "allege a massive cover-up that will destroy everything you thought was true",
]

real_templates = [
    "{subj} {act}. Officials say further details will be shared in the coming weeks after review by relevant committees.",
    "{subj} {act}, according to a report released on Monday. The findings are consistent with previous data.",
    "In a statement on Tuesday, {subj} {act}. Analysts note the move was widely expected.",
    "{subj} {act} following months of deliberation. The decision will be reviewed again next quarter.",
]

fake_templates = [
    "You won't BELIEVE what {subj} just did! {act_cap}, and the mainstream media refuses to report it!!!",
    "SHOCKING: {subj} {act}. Share this before it gets DELETED by the deep state!",
    "BREAKING (they don't want you to see this): {subj} {act}. Wake up, sheeple!",
    "This will blow your mind — {subj} {act}. Doctors and governments HATE this one simple fact!",
]


def make_row(is_real: bool):
    if is_real:
        subj = random.choice(real_subjects)
        act = random.choice(real_actions)
        template = random.choice(real_templates)
        text = template.format(subj=subj, act=act)
        label = "REAL"
    else:
        subj = random.choice(fake_subjects)
        act = random.choice(fake_actions)
        template = random.choice(fake_templates)
        text = template.format(subj=subj, act=act, act_cap=act.capitalize())
        label = "FAKE"
    return text, label


def main(n_per_class: int = 150, out_path: str = "sample_data.csv"):
    rows = []
    for _ in range(n_per_class):
        rows.append(make_row(True))
        rows.append(make_row(False))
    random.shuffle(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
