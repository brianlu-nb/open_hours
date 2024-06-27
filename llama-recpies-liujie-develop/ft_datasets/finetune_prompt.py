
writing_quality_prompt = '''We measure an article's writing quality based on the following 
Given a article title and content below, please evaluate its writing quality based on the following 8 dimensions:
1. Clarity and Coherence: The ability of the article to communicate ideas in a clear and logical manner. This includes the use of straightforward language, well-structured sentences, and a coherent flow of ideas.
2. Relevance and Focus: The article should stay focused on the central topic or argument without diverging into unrelated areas. Relevance ensures that all parts of the article contribute directly to its main purpose or thesis.
3. Accuracy and Credibility: Factual accuracy is crucial for building trust with readers. This involves verifying facts, citing reliable sources, and providing accurate representations of data and events.
4. Originality and Insightfulness: Original content that provides new insights, perspectives, or findings adds significant value. This dimension assesses the uniqueness of the article and its contribution to the subject matter.
5. Engagement and Interest: The ability to capture and hold the reader's attention through compelling writing, interesting anecdotes, or thought-provoking questions. Engaging writing often includes a strong introduction, dynamic content, and a memorable conclusion.
6. Grammar and Style: Proper grammar, punctuation, and spelling are fundamental to writing quality. Style also plays a role, including the choice of words, tone, and the use of literary devices that enhance the reading experience.
7. Structural Integrity: The organization of the article, including the use of headings, subheadings, paragraphs, and bullet points, which can help in making the content more digestible and easier to follow.
8. Argumentation and Evidence: For articles that aim to persuade or present an argument, the quality of reasoning, the strength of evidence provided, and the effectiveness of the argumentative structure are important. 
Below is an article title and content:
Title: {title}
Content: {content}
Please evaluate its write quality, assign a score (1 to 5) for each dimension, and generate a rationale.
Then generate a overall quality score (1 to 5) and a rationale.
Please output with the following json format 
{{"Clarity_and_Coherence_Score": score (1 to 5), "Clarity_and_Coherence_Evaluation_Rationale": XXX, 
"Relevance_and_Focus_Score": score (1 to 5), "Relevance_and_Focus_Evaluation_Rationale": XXX, 
"Accuracy_and_Credibility_Score": score (1 to 5), "Accuracy_and_Credibility_Evaluation_Rationale": XXX, 
"Originality_and_Insightfulness_Score": score (1 to 5), "Originality_and_Insightfulness_Evaluation_Rationale": XXX, 
"Engagement_and_Interest_Score": score (1 to 5), "Engagement_and_Interest_Evaluation_Rationale": XXX, 
"Grammar_and_Style_Score": score (1 to 5), "Grammar_and_Style_Evaluation_Rationale": XXX, 
"Structural_Integrity_Score": score (1 to 5), "Structural_Integrity_Evaluation_Rationale": XXX, 
"Argumentation_and_Evidence_Score": score (1 to 5), "Argumentation_and_Evidence_Evaluation_Rationale": XXX, 
"Overall_Score": score (1 to 5), "Overall_Evaluation_Rationale": XXX}}
Please output now:
'''

writing_quality_without_rationale_prompt = '''We measure an article's writing quality based on the following 
Given a article title and content below, please evaluate its writing quality based on the following 8 dimensions:
1. Clarity and Coherence: The ability of the article to communicate ideas in a clear and logical manner. This includes the use of straightforward language, well-structured sentences, and a coherent flow of ideas.
2. Relevance and Focus: The article should stay focused on the central topic or argument without diverging into unrelated areas. Relevance ensures that all parts of the article contribute directly to its main purpose or thesis.
3. Accuracy and Credibility: Factual accuracy is crucial for building trust with readers. This involves verifying facts, citing reliable sources, and providing accurate representations of data and events.
4. Originality and Insightfulness: Original content that provides new insights, perspectives, or findings adds significant value. This dimension assesses the uniqueness of the article and its contribution to the subject matter.
5. Engagement and Interest: The ability to capture and hold the reader's attention through compelling writing, interesting anecdotes, or thought-provoking questions. Engaging writing often includes a strong introduction, dynamic content, and a memorable conclusion.
6. Grammar and Style: Proper grammar, punctuation, and spelling are fundamental to writing quality. Style also plays a role, including the choice of words, tone, and the use of literary devices that enhance the reading experience.
7. Structural Integrity: The organization of the article, including the use of headings, subheadings, paragraphs, and bullet points, which can help in making the content more digestible and easier to follow.
8. Argumentation and Evidence: For articles that aim to persuade or present an argument, the quality of reasoning, the strength of evidence provided, and the effectiveness of the argumentative structure are important. 
Below is an article title and content:
Title: {title}
Content: {content}
Please evaluate its write quality, assign a score (1 to 5) for each dimension.
Then generate a overall quality score (1 to 5).
Please output with the following json format 
{{"Clarity_and_Coherence_Score": score (1 to 5), 
"Relevance_and_Focus_Score": score (1 to 5), 
"Accuracy_and_Credibility_Score": score (1 to 5), 
"Originality_and_Insightfulness_Score": score (1 to 5), 
"Engagement_and_Interest_Score": score (1 to 5), 
"Grammar_and_Style_Score": score (1 to 5), 
"Structural_Integrity_Score": score (1 to 5), 
"Argumentation_and_Evidence_Score": score (1 to 5), 
"Overall_Score": score (1 to 5)}}
Please output now:
'''

content_genre_template_bak = '''
You are an content editor to judge the content category of given articles.
Given the title and content of an article, please tell the genre of the article, please choose one of the following 16 genre categories:
1. News: Brief, factual reports on current events. These articles are timely, objective, and focus on delivering the who, what, when, where, why, and how of an incident or development.
2. Opinion: Articles expressing the personal views of the author on a topic, aiming to persuade or influence the reader's thoughts.
3. Feature: In-depth articles providing a comprehensive exploration of a topic, celebrity person, or event, often with narrative storytelling elements.
4. Editorial: Opinion pieces representing the official stance of the publication, reflecting the collective view of its leadership on current issues.
5. Column: Regularly appearing articles by the same author offering opinions or commentary on a particular field, characterized by a personal style.
6. PersonStory: News report that focuses on an individual or small group's experiences, challenges, or achievements. These stories are typically designed to evoke emotion, inspire, or provoke thought in the reader.
7. Review: Critical evaluations of products, services, or creative works such as movies, books, or music, offering insights into their quality and value.
8. Interview: A conversation between the interviewer and interviewee, providing insights into the interviewee's perspectives, experiences, or expertise.
9. Guide: Instructional content offering step-by-step advice or guidelines on how to perform tasks, solve problems, or accomplish specific objectives.
10. Analysis: Detailed examinations of topics or data, often incorporating studies, surveys, or expert insights to provide a deeper understanding or new findings.
11. Essay: A short piece of writing on a particular subject, blending facts with the author's personal reflections or arguments.
12. Story: Narrative works focusing on characters and events, either fictional or based on real-life, crafted to entertain, inform, or convey messages
13. Investigative: In-depth reporting aiming to uncover the truth about complex issues, often involving extensive research and the examination of hidden or contentious matters.
14. Profile: Comprehensive portrayals of individuals, highlighting their life, achievements, and significance within a particular context.
15. Obituary: Notices of a person's death, summarizing their life, legacy, and impact, often including details about surviving family and funeral arrangements.
16. Satire: Content using humor, irony, exaggeration, or ridicule to critique or mock subjects, often relating to current events, societal norms, or human behavior.
Below is an article title and content:
Title: {title}
Content: {content}
Please output with the following json format
{{"genre": XXX, "reason": XXX}}
Please output now:
'''

content_genre_template = '''
You are an content editor to judge the content category of given articles.
Given the title and content of an article, please tell the category of the article, please choose one of the following 10 content categories:
1. News: News is a journalistic account of current events and developments. These stories aim to answer the questions of who, what, where, when, why, presenting information in a clear and concise and factual manner without editorializing or interpretation. The guiding principles governing the writing of news include: accuracy, objectivity, independence, fairness, and balance. News stories span a broad spectrum of topics of public interest, including political events, natural disasters, crime reports, and economic trends, and more.
2. Investigative: Investigative articles involve in-depth research and reporting to uncover hidden information, expose wrongdoing, or shed light on complex issues, such as corruption, corporate malfeasance, government surveillance, and social injustice. They often require extensive time and resources and result in significant revelations or impact. Comparing the writing style to hard news, investigative journalism often involves a more narrative-driven approach to storytelling, incorporating elements of suspense, intrigue, and revelation.
3. Opinion: This category encompasses a variety of content types where the author expresses their viewpoint or perspective on a particular topic.Editorials are written by the editorial board or editorial staff of the publication.Opinion, or op-ed, are authored by experts, columnists, community members, or regular citizens, who are individuals who are not affiliated with the publication.Letters to the Editor is a section commonly found in newspapers, magazines, and online publications where readers can express their responses to articles published by the publication, commentary on current events, personal anecdotes, suggestions for improvement, or expressions of gratitude.
4. Interview: Interview articles are journalistic pieces that present information, insights, or opinions obtained through interviews with individuals or groups. Q&A format includes each question and its corresponding answers presented as separate paragraphs or sections. And in narrative interview articles, the interviewer's questions and the interviewee's responses are integrated into a cohesive narrative framework.
5. Profile: Profile and obituary articles provide a comprehensive overview of an individual's life, achievements, and legacy, allowing readers to connect on a personal level. Profile articles may cover a wide range of individuals, including public figures, celebrities, leaders, activists, artists, entrepreneurs, or ordinary people with compelling stories. While they may be influenced by current events, the focus is usually not on reporting recent developments.
6. Review: Review articles evaluate and critique various forms of media, products, services, or events. They provide an assessment of quality, performance, or relevance based on specific criteria, offering guidance to potential consumers or audiences.
7. Guide: How-to or guide articles offer step-by-step instructions or advice on accomplishing a wide variety of tasks, from DIY projects and technical skills to health and wellness practices and career development strategies. They are typically informative and instructional in nature, aiming to help readers achieve specific objectives.
8. Analysis: Research or analysis articles delve into a particular topic or issue, often involving a comprehensive examination of data, evidence, or theories.They are widespread across various domains, including business and economics, technology and innovation, healthcare and medicine, environmental studies. These pieces aim to inform readers about the findings or implications of the research conducted. 
9, Essay: Essay or story articles present narrative-driven content, exploring themes, ideas, or experiences through storytelling. They can range from personal anecdotes to fictional narratives. Essay and story pieces are generally not tied to timeliness in the same way that news articles are. While they may be influenced by current events or personal experiences, their primary focus is often on exploring broader themes。
10, Satire: Satire serves as a form of commentary that blends elements of humor, irony, parody, and  juxtaposition with factual reporting to critique policies, events, and societal norms, or to highlight the hypocrisy, incompetence, or folly of political figures. Satirical news articles often feature attention-grabbing headlines or titles that mimic the style of traditional news headlines. Satirical articles may offer humorous yet insightful analysis of complex political topics, providing audiences with alternative perspectives or highlighting the absurdity of certain arguments or positions. Humor or satire articles usually clearly label themselves as such, but there may be instances where the labeling is less explicit or even absent.
Below is an article title and content:
Title: {title}
Content: {content}
Please output with the following json format
{{"category": XXX, "reason": XXX}}
Please output now:
'''

content_genre_without_reason_template = '''
You are an content editor to judge the content category of given articles.
Given the title and content of an article, please tell the category of the article, please choose one of the following 10 content categories:
1. News: News is a journalistic account of current events and developments. These stories aim to answer the questions of who, what, where, when, why, presenting information in a clear and concise and factual manner without editorializing or interpretation. The guiding principles governing the writing of news include: accuracy, objectivity, independence, fairness, and balance. News stories span a broad spectrum of topics of public interest, including political events, natural disasters, crime reports, and economic trends, and more.
2. Investigative: Investigative articles involve in-depth research and reporting to uncover hidden information, expose wrongdoing, or shed light on complex issues, such as corruption, corporate malfeasance, government surveillance, and social injustice. They often require extensive time and resources and result in significant revelations or impact. Comparing the writing style to hard news, investigative journalism often involves a more narrative-driven approach to storytelling, incorporating elements of suspense, intrigue, and revelation.
3. Opinion: This category encompasses a variety of content types where the author expresses their viewpoint or perspective on a particular topic.Editorials are written by the editorial board or editorial staff of the publication.Opinion, or op-ed, are authored by experts, columnists, community members, or regular citizens, who are individuals who are not affiliated with the publication.Letters to the Editor is a section commonly found in newspapers, magazines, and online publications where readers can express their responses to articles published by the publication, commentary on current events, personal anecdotes, suggestions for improvement, or expressions of gratitude.
4. Interview: Interview articles are journalistic pieces that present information, insights, or opinions obtained through interviews with individuals or groups. Q&A format includes each question and its corresponding answers presented as separate paragraphs or sections. And in narrative interview articles, the interviewer's questions and the interviewee's responses are integrated into a cohesive narrative framework.
5. Profile: Profile and obituary articles provide a comprehensive overview of an individual's life, achievements, and legacy, allowing readers to connect on a personal level. Profile articles may cover a wide range of individuals, including public figures, celebrities, leaders, activists, artists, entrepreneurs, or ordinary people with compelling stories. While they may be influenced by current events, the focus is usually not on reporting recent developments.
6. Review: Review articles evaluate and critique various forms of media, products, services, or events. They provide an assessment of quality, performance, or relevance based on specific criteria, offering guidance to potential consumers or audiences.
7. Guide: How-to or guide articles offer step-by-step instructions or advice on accomplishing a wide variety of tasks, from DIY projects and technical skills to health and wellness practices and career development strategies. They are typically informative and instructional in nature, aiming to help readers achieve specific objectives.
8. Analysis: Research or analysis articles delve into a particular topic or issue, often involving a comprehensive examination of data, evidence, or theories.They are widespread across various domains, including business and economics, technology and innovation, healthcare and medicine, environmental studies. These pieces aim to inform readers about the findings or implications of the research conducted. 
9, Essay: Essay or story articles present narrative-driven content, exploring themes, ideas, or experiences through storytelling. They can range from personal anecdotes to fictional narratives. Essay and story pieces are generally not tied to timeliness in the same way that news articles are. While they may be influenced by current events or personal experiences, their primary focus is often on exploring broader themes。
10, Satire: Satire serves as a form of commentary that blends elements of humor, irony, parody, and  juxtaposition with factual reporting to critique policies, events, and societal norms, or to highlight the hypocrisy, incompetence, or folly of political figures. Satirical news articles often feature attention-grabbing headlines or titles that mimic the style of traditional news headlines. Satirical articles may offer humorous yet insightful analysis of complex political topics, providing audiences with alternative perspectives or highlighting the absurdity of certain arguments or positions. Humor or satire articles usually clearly label themselves as such, but there may be instances where the labeling is less explicit or even absent.
Below is an article title and content:
Title: {title}
Content: {content}
Please output with the following json format
{{"category": XXX}}
Please output now:
'''