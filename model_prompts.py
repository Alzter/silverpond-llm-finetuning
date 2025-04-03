DBPEDIA = {
    # Zero-shot prompt to classify articles and return their category name.
    ZERO_SHOT : """You are an expert at classifying articles into the following categories:

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
    ZERO_SHOT : """You are an expert at classifying OSHA injury reports into the following categories:

    CATEGORIES:
    0. Abrasions, scratches
    1. Amputations
    2. Amputations involving bone loss
    3. Amputations, avulsions, enucleations  unspecified
    4. Amputations, avulsions, enucleations, n.e.c.
    5. Anaphylactic shock, anaphylaxis
    6. Anxiety, stress  unspecified
    7. Asphyxiations, strangulations, suffocations
    8. Avulsions, enucleations
    9. Avulsions, enucleations without bone loss
    10. Blisters
    11. Bruises, contusions
    12. Burns and corrosions, unspecified
    13. Burns and other injuries, n.e.c.
    14. Caisson disease, bends, divers' palsy
    15. Cerebral and other intracranial hemorrhages
    16. Cerebral and other intracranial hemorrhages without skull fracture
    17. Chemical burns and corrosions, unspecified
    18. Chemical burns, corrosions  degree unspecified
    19. Circulatory system diseases, unspecified
    20. Concussions
    21. Convulsions, seizures
    22. Coughing and throat irritation- toxic, noxious, or allergenic effect
    23. Crushing injuries
    24. Cuts and abrasions or bruises
    25. Cuts, lacerations
    26. Cuts, lacerations, punctures without injury to internal structures
    27. Damage to artificial limb(s)
    28. Damage to medical implants, n.e.c.
    29. Dermatitis and reactions affecting the skin-acute, unspecified
    30. Disc disorders, herniated disc
    31. Dislocation of joints
    32. Dislocations
    33. Dislocations, n.e.c.
    34. Dislocations, unspecified
    35. Disorders of the ear, mastoid process, hearing, unspecified
    36. Dizziness, lightheadedness, headache-toxic, noxious, or allergenic effect
    37. EXPOSURES TO DISEASE-NO ILLNESS INCURRED
    38. Effects of heat  n.e.c.
    39. Effects of heat  unspecified
    40. Effects of heat and light, n.e.c.
    41. Effects of heat and light, unspecified
    42. Effects of poison, toxic, or allergenic exposure  n.e.c.
    43. Effects of poison, toxic, or allergenic exposure  unspecified
    44. Effects of reduced temperature, n.e.c.
    45. Electrical burns  any degree
    46. Electrical burns, unspecified
    47. Electrocution, electric shock
    48. Electrocutions, electric shocks
    49. First degree chemical burns and corrosions
    50. First degree electrical burns
    51. First degree heat (thermal) burns
    52. Fractures
    53. Fractures (except rib, trunk fractures) and internal injuries
    54. Fractures (except skull fractures) and concussions
    55. Fractures and burns
    56. Fractures and dislocations
    57. Fractures and other injuries, n.e.c.
    58. Fractures and other injuries, unspecified
    59. Fractures and soft tissue injuries
    60. Fractures and surface, flesh wounds
    61. Frostbite
    62. General symptoms  unspecified
    63. Gunshot wounds
    64. Heat (thermal) burns, unspecified
    65. Heat exhaustion, fatigue
    66. Heat exhaustion, prostration
    67. Heat stroke
    68. Heat stroke, syncope
    69. Heat syncope
    70. Hernias
    71. Hernias due to traumatic incidents
    72. Herniated discs
    73. Hyperventilation
    74. Injuries to internal organs, major blood vessels  unspecified
    75. Injuries to the brain, spinal cord and severe wounds, internal injuries
    76. Internal injuries to organs and blood vessels of the trunk
    77. Intracranial injuries  unspecified
    78. Intracranial injuries and injuries to internal organs
    79. Intracranial injuries with skull fractures
    80. Intracranial injuries, n.e.c.
    81. Intracranial injuries, unspecified
    82. Irritant dermatitis-acute
    83. Ischemic heart disease, unspecified
    84. Loss of consciousness-not heat related
    85. MULTIPLE DISEASES, CONDITIONS, AND DISORDERS
    86. Major tears to muscles, tendons, ligaments
    87. Multiple effects of heat and light
    88. Multiple intracranial injuries, n.e.c.
    89. Multiple intracranial injuries, unspecified
    90. Multiple nonspecified injuries and disorders
    91. Multiple poisoning, toxic, noxious, or allergenic effects
    92. Multiple severe wounds and internal injuries
    93. Multiple sprains, strains, tears
    94. Multiple surface and flesh wounds
    95. Multiple surface wounds and bruises
    96. Multiple symptoms
    97. Multiple traumatic injuries and disorders  unspecified
    98. Multiple traumatic injuries and disorders, n.e.c.
    99. Multiple traumatic injuries and disorders, unspecified
    100. Multiple traumatic injuries to muscles, tendons, ligaments, joints, etc.
    101. Multiple types of dislocations
    102. Multiple types of open wounds
    103. Myocardial infarction (heart attack)
    104. Nausea, vomiting- toxic, noxious, or allergenic effect
    105. Nonfatal 'crushing' injuries
    106. Nonspecified injuries and disorders, n.e.c.
    107. Open wounds involving internal organs, major blood vessels
    108. Open wounds, unspecified
    109. Other burns, second degree
    110. Other burns, unspecified
    111. Other multiple traumatic injuries  n.e.c.
    112. Other or unspecified allergic reactions
    113. Other poisoning, toxic, noxious, or allergenic effects, n.e.c.
    114. Other respiratory system symptoms-toxic, noxious, or allergenic effect
    115. Other traumatic injuries and disorders, unspecified
    116. Paralysis, paraplegia, quadriplegia
    117. Pinched nerve
    118. Poisoning, including poisoning-related asphyxia
    119. Poisoning, poisoning-related asphyxia
    120. Poisoning, toxic, noxious, or allergenic effect, unspecified
    121. Pulmonary edema
    122. Puncture wounds, except gunshot wounds
    123. Second degree chemical burns and corrosions
    124. Second degree electrical burns
    125. Second degree heat (thermal) burns
    126. Severe wounds, internal injuries and electrocution, electric shock
    127. Skull fracture and intracranial injury
    128. Soft tissue injuries  unspecified
    129. Soreness, pain, hurt-nonspecified injury
    130. Soreness, swelling, inflammation
    131. Sprains
    132. Sprains and cuts
    133. Sprains, strains, minor tears
    134. Sprains, strains, tears  unspecified
    135. Sprains, strains, tears, unspecified
    136. Strains
    137. Stroke
    138. Surface, flesh wounds and burns, electrical injuries
    139. Swelling, inflammation, irritation-nonspecified injury
    140. Thermal burns  degree unspecified
    141. Thermal burns  second degree
    142. Thermal burns  third degree or higher
    143. Third or fourth degree chemical burns and corrosions
    144. Third or fourth degree electrical burns
    145. Third or fourth degree heat (thermal) burns
    146. Traumatic injuries and disorders, n.e.c.
    147. Traumatic injuries and disorders, unspecified
    148. Traumatic injuries or exposures  unspecified
    149. Traumatic injuries to bones, nerves, spinal cord, unspecified
    150. Traumatic injuries to muscles, tendons, ligaments, joints, etc., n.e.c.
    151. Traumatic injuries to muscles, tendons, ligaments, joints, etc., unspecified
    152. Traumatic injuries to nerves, except the spinal cord, n.e.c.
    153. Traumatic injuries to nerves, except the spinal cord, unspecified
    154. Traumatic injuries to spinal cord, n.e.c.
    155. Traumatic injuries to spinal cord, unspecified
    156. Whiplash

    Read the following OSHA injury report then answer with the name of the category which suits it best.
    Answer with ONLY the name of the category, i.e. "Sprains and cuts"."""
}
