---
task_categories:
- visual-question-answering
language:
- en
size_categories:
- 1K<n<10K
---
# Dataset Card for BloomVQA


### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

BloomVQA is a dataset based on picture stories designed for educating young children. 
It aims to facilitate comprehensive evaluation and characterization of vision-language models on comprehension tasks. 
The dataset contains tasks reflecting 6 different levels of comprehension and underlying cognitive processes, 
as laid out in Bloom's Taxonomy, a classic framework widely adopted in education research. 
This underlying hierarchical taxonomy enables graded model evaluation, automatic data augmentation and novel metrics characterizing model consistency. 

The core dataset contains 1200 multiple-choice samples collected via Amazon Mechanical Turk based on 20 picture stories downloaded from Creative Commons resources [Book Dash](https://bookdash.org/) and [Storyweaver](https://storyweaver.org.in/en/).

<!-- Provide the basic links for the dataset. -->

- **Paper:** [BloomVQA: Assessing Hierarchical Multi-modal Comprehension](https://arxiv.org/abs/2312.12716)



## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

Each multiple-choice sample contains 1 question and 4 free-form answers including 1 correct answer and 3 incorrect answers. Each sample is labeled with the title of picture story and the level of comprehension as defined in Bloom's Taxonomy.
