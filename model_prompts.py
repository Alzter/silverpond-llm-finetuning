# Zero-shot prompt to classify articles and return their category name.
PROMPT_ZEROSHOT = """You are an expert at classifying articles into the following categories:

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
Answer with ONLY the name of the category, i.e. "Company".
"""

# Zero-shot chain-of-thought prompt to classify articles and return their category name.
PROMPT_COT = """You are an expert at classifying articles into the following categories:

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

Read the following article and explain which category describes its content best.
End your answer with the category name, i.e. "Company".
Let's think step by step.
"""

# Meta prompt - A type of zero-shot prompt which prioritises abstract reasoning over concrete examples.
# Advantages: Fewer tokens. Disadvantages:
# https://www.promptingguide.ai/techniques/meta-prompting
PROMPT_META = """Problem: [excerpt from an encyclopedia article]

Solution Structure:
1. Begin the response with "Let's think step by step".
2. Identify the subject of the encyclopedia article with "This encyclopedia article is about [subject]".
3. Define what the subject is. Is it natural or artificial? Is it one or multiple entities? Use "[subject] is a [classification]".
4. Consider the following list of categories:
    - Company
    - EducationalInstitution
    - Artist
    - Athlete
    - OfficeHolder
    - MeanOfTransportation
    - Building
    - NaturalPlace
    - Village
    - Animal
    - Plant
    - Album
    - Film
    - WrittenWork
   Identify all categories in this list whose properties do not match the subject.
5. Identify which category has the most in common with the subject and explain why.
6. Finally, state "Category: [best matching category]."
"""

# Few-shot chain-of-thought prompt to classify articles and return their category name.
PROMPT_COT_4SHOT = """You are an expert at classifying articles into the following categories:

CATEGORIES:
- Company
- EducationalInstitution
- Artist
- Athlete
- OfficeHolder
- MeanOfTransportation
- Building
- NaturalPlace
- Village
- Animal
- Plant
- Album
- Film
- WrittenWork

Problem:
The Petlyakov VI-100 (Visotnyi Istrebitel – high altitude fighter) was a fighter/dive bomber aircraft designed and built in the USSR from 1938.

Solution:
Let's think step by step. This encyclopedia article is about the Petlyakov VI-100, which is an aircraft. While aircrafts are a man-made structure designed and built for a specific purpose, they are not human habitats with walls and a ceiling, so Building is not the category. Aircrafts are designed to transport people, so MeanOfTransportation is the best category. Category: MeanOfTransportation.

Problem:
Kruszewo [kruˈʂɛvɔ] is a village in the administrative district of Gmina Żuromin within Żuromin County Masovian Voivodeship in east-central Poland.

Solution:
Let's think step by step. This encyclopedia article is about Kruszewo, which is a village in Poland. While villages do exist within geographical areas, they are man-made, so NaturalPlace is not the category. While villages do contain buildings, they are not a single building, so Building is not the category. The most matching category is therefore Village. Category: Village.

Problem:
Schismus is a genus of grass in the Poaceae family. They are native to Africa and Asia.

Solution:
Let's think step by step. This encyclopedia article is about Schismus, which is a biological species. The genus of the species is grass. Grass is commonly found in natural places, but Schismus is not a geographical location, so NaturalPlace is not the category. Grass is a type of plant, so Plant is the most fitting category. Category: Plant.

Problem:
The Southern Oklahoma Cosmic Trigger Contest is a soundtrack by The Flaming Lips to the Bradley Beesley fishing documentary Okie Noodling.

Solution:
Let's think step by step. This encyclopedia article is about The Southern Oklahoma Cosmic Trigger Contest, which is a soundtrack to a fishing documentary. While the article mentions a fishing documentary, it is not the subject, so Film is not the category. While the article mentions the band The Flaming Lips, they are not the subject, so Artist is not the category. The most suitable category is therefore Album. Category: Album.

Problem:
"""

# Few-shot chain-of-thought prompt to classify articles and return their category name.
PROMPT_COT_2SHOT = """You are an expert at classifying articles into the following categories:

CATEGORIES:
- Company
- EducationalInstitution
- Artist
- Athlete
- OfficeHolder
- MeanOfTransportation
- Building
- NaturalPlace
- Village
- Animal
- Plant
- Album
- Film
- WrittenWork

Problem:
The Petlyakov VI-100 (Visotnyi Istrebitel – high altitude fighter) was a fighter/dive bomber aircraft designed and built in the USSR from 1938.

Solution:
Let's think step by step. This encyclopedia article is about the Petlyakov VI-100, which is an aircraft. While aircrafts are a man-made structure designed and built for a specific purpose, they are not human habitats with walls and a ceiling, so Building is not the category. Aircrafts are designed to transport people, so MeanOfTransportation is the best category. Category: MeanOfTransportation.

Problem:
Schismus is a genus of grass in the Poaceae family. They are native to Africa and Asia.

Solution:
Let's think step by step. This encyclopedia article is about Schismus, which is a biological species. The genus of the species is grass. Grass is commonly found in natural places, but Schismus is not a geographical location, so NaturalPlace is not the category. Grass is a type of plant, so Plant is the most fitting category. Category: Plant.

Problem:
"""