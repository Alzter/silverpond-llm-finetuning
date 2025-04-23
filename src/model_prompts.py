DBPEDIA = {
    # Zero-shot prompt to classify articles and return their category name.
    "ZERO_SHOT" : """You are an expert at classifying articles into the following categories:

    CATEGORIES:
    0. Company
    1. EducationalInstitution
    2. Artist
    3. Athlete
    4. OfficeHolder
    5. MeanOfTransportation
    6. Building
    7. NaturalPlace
    8. Village
    9. Animal
    10. Plant
    11. Album
    12. Film
    13. WrittenWork

    Read the following article then answer with the name of the category which suits it best.
    Answer with ONLY the name of the category, i.e. "Company"."""
}

OSHA = {
    "ZERO_SHOT" : """You are an expert at classifying OSHA injury reports into the following categories:

    CATEGORIES:
    0. Amputations
    1. Crushing injuries
    2. Cuts, lacerations
    3. Fractures
    4. Heat (thermal) burns, unspecified
    5. Internal injuries to organs and blood vessels of the trunk
    6. Intracranial injuries, unspecified
    7. Puncture wounds, except gunshot wounds
    8. Soreness, pain, hurt-nonspecified injury
    9. Traumatic injuries and disorders, unspecified

    Read the following OSHA injury report then answer with the name of the category which suits it best.
    Answer with ONLY the name of the category, i.e. "Fractures"."""
}
OSHA = {
    "ZERO_SHOT" : """You are an expert at classifying OSHA injury reports into the following categories:

    CATEGORIES:
    0. Amputations
    1. Crushing injuries
    2. Cuts, lacerations
    3. Fractures
    4. Heat (thermal) burns, unspecified
    5. Internal injuries to organs and blood vessels of the trunk
    6. Intracranial injuries, unspecified
    7. Puncture wounds, except gunshot wounds
    8. Soreness, pain, hurt-nonspecified injury
    9. Traumatic injuries and disorders, unspecified

    Read the following OSHA injury report then answer with the name of the category which suits it best.
    Answer with ONLY the name of the category, i.e. "Fractures".""",

    "MULTI_TASK" : """You are an expert at classifying OSHA injury reports.

    You are given two classification tasks.
    You should output the result as a two json fields as {"NatureTitle": "nature_title_label", "Part of Body Title": "part_of_body_title_label"}
    For NatureTitle, given the injury report, you are asked to classify it as one of the labels in the list ["Amputations", "Crushing injuries", "Cuts, lacerations", "Fractures", "Heat (thermal) burns, unspecified", "Internal injuries to organs and blood vessels of the trunk", "Intracranial injuries, unspecified", "Puncture wounds, except gunshot wounds", "Soreness, pain, hurt-nonspecified injury", "Traumatic injuries and disorders, unspecified"] and change nature_title_label to the correct label in the list.
    For Part of Body Title, given the injury report, you are asked to classify it as one of the labels in the list ["Ankle(s)", "BODY SYSTEMS", "Brain", "Finger(s), fingernail(s), n.e.c.", "Finger(s), fingernail(s), unspecified", "Fingertip(s)", "Hip(s)", "Leg(s), unspecified", "Multiple body parts, n.e.c.", "Nonclassifiable"] and change part_of_body_title_label to the correct label in the list.

    Output the two json fields only and absolutely nothing else.
    Now it is your turn.
    """
}
