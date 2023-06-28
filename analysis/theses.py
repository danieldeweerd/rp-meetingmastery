import os.path

import numpy as np

from analysis.ai_utils import get_minilm_embedding_batch, get_minilm_embedding

argumentative_theses = [
    "Veganism is a more ethical and sustainable dietary choice than consuming animal products.",
    "Social media has a detrimental impact on mental health and overall well-being.",
    "The death penalty should be abolished as it violates the right to life.",
    "Climate change is primarily caused by human activities and requires urgent global action.",
    "The use of genetically modified organisms (GMOs) in food production is safe and beneficial.",
    "Free college education should be provided to all students to promote equal opportunities.",
    "The legalization of marijuana has numerous medical and economic benefits.",
    "Capital punishment is an effective deterrent against serious crimes.",
    "The minimum legal drinking age should be lowered to 18.",
    "Animal testing is necessary for scientific and medical advancements.",
    "Universal healthcare is a fundamental right that should be guaranteed by governments.",
    "Technology has a negative impact on interpersonal communication skills.",
    "The United Nations plays a crucial role in maintaining global peace and security.",
    "Standardized tests are an accurate measure of a student's knowledge and abilities.",
    "Nuclear power is a safe and clean alternative to fossil fuels.",
    "Censorship is necessary to protect society from offensive and harmful content.",
    "Affirmative action is necessary to address historical inequalities and promote diversity.",
    "Genetic engineering has the potential to solve major world problems.",
    "Online dating is a viable and effective way to meet potential partners.",
    "The government should impose higher taxes on sugary drinks to reduce obesity rates.",
    "Gun control laws should be stricter to prevent mass shootings.",
    "Artificial intelligence will eventually surpass human intelligence and pose risks.",
    "Parents should be held legally responsible for their children's crimes.",
    "The use of cell phones while driving should be completely banned.",
    "Space exploration is a worthwhile investment for scientific discovery.",
    "The traditional book format is superior to e-books in preserving knowledge and culture.",
    "School uniforms promote discipline and enhance the learning environment.",
    "The entertainment industry has a responsibility to represent diverse voices and stories.",
    "Euthanasia should be legalized as an option for terminally ill patients.",
    "Public transportation is a more sustainable and efficient mode of travel than private cars.",
    "Video games contribute to increased aggression and violence in young people.",
    "Organ donation should be mandatory for everyone.",
    "The internet has improved access to information and fostered global connectivity.",
    "Banning plastic bags is necessary to reduce environmental pollution.",
    "Parents should have the right to homeschool their children.",
    "Foreign aid is essential in addressing global poverty and inequality.",
    "Sex education should be comprehensive and mandatory in all schools.",
    "Art and music programs are important for the overall development of students.",
    "Preserving endangered species is crucial for maintaining biodiversity.",
    "Space exploration is a waste of resources and should be discontinued.",
    "Human cloning should be prohibited due to ethical and moral concerns.",
    "The use of alternative energy sources is key to mitigating climate change.",
    "The government should provide more support for renewable energy initiatives.",
    "Fast food should display calorie information to promote healthier eating habits.",
    "Social media platforms should be held accountable for moderating harmful content.",
    "Parents should limit the amount of time their children spend on electronic devices.",
    "The education system should focus more on practical skills and real-world applications.",
    "Zoos and aquariums contribute to the conservation of endangered species.",
    "Foreign language education should start in elementary school.",
    "Artificial intelligence will lead to widespread job displacement and unemployment.",
    "Public smoking should be banned to protect the health of individuals and reduce the burden on healthcare systems.",
    "The government should provide financial incentives for businesses to implement sustainable practices.",
    "Mandatory voting would improve democratic participation and representation.",
    "Schools should teach financial literacy to prepare students for managing personal finances.",
    "Organic farming is a more environmentally friendly and healthier approach to agriculture.",
    "The use of animals for entertainment purposes, such as circuses or rodeos, should be prohibited.",
    "Standardized testing puts too much emphasis on memorization and fails to assess critical thinking skills.",
    "The death penalty does not effectively deter crime and risks executing innocent individuals.",
    "The internet has facilitated the spread of fake news and misinformation, leading to societal harm.",
    "Government surveillance programs pose a threat to individual privacy rights.",
    "The benefits of nuclear energy outweigh the potential risks associated with accidents.",
    "The legal drinking age should remain at 21 to protect the well-being of young people.",
    "Fast fashion has negative environmental and ethical consequences and should be discouraged.",
    "Parents should have the right to choose their child's education, including homeschooling or alternative schooling methods.",
    "The use of performance-enhancing drugs in sports should result in severe penalties.",
    "The electoral college should be abolished in favor of a popular vote system.",
    "Artificial intelligence and automation will lead to job creation and new opportunities.",
    "Mandatory military service promotes civic duty and fosters a sense of national unity.",
    "Prisons should focus on rehabilitation rather than punishment to reduce recidivism rates.",
    "Privacy is a fundamental right that should be protected, even in the digital age.",
    "Alternative medicine should be subject to the same scientific scrutiny as conventional medicine.",
    "The use of drones for targeted killings violates international law and ethical principles.",
    "Parental consent should be required for teenagers to undergo cosmetic procedures.",
    "Multinational corporations have a responsibility to prioritize environmental sustainability over profits.",
    "The use of animals in scientific research is necessary for medical advancements and human welfare.",
    "Religion should not play a role in political decision-making and governance.",
    "Mandatory voting would ensure a more representative and inclusive democracy.",
    "Artificial intelligence poses significant ethical challenges, including the potential for bias and discrimination.",
    "The consumption of sugary drinks should be heavily taxed to combat rising obesity rates.",
    "The government should invest more in renewable energy research and development.",
    "The cultural appropriation of marginalized cultures is offensive and disrespectful.",
    "College athletes should be compensated for their participation in revenue-generating sports.",
    "Sustainable development should be a top priority in urban planning and infrastructure projects.",
    "Social media platforms should take stronger measures to combat cyberbullying and harassment.",
    "Government surveillance is necessary to prevent terrorism and protect national security.",
    "Animal agriculture is a major contributor to climate change and should be reduced.",
    "The use of plastic should be minimized to protect the environment and marine life.",
    "The benefits of nuclear energy outweigh the risks, as modern technology ensures safety.",
    "Parents should be required to vaccinate their children to protect public health.",
    "Education should prioritize teaching critical thinking skills and problem-solving abilities.",
    "The government should provide subsidies for electric vehicles to promote sustainable transportation.",
    "Artificial intelligence will create more jobs than it displaces in the long run.",
    "Paid maternity and paternity leave should be guaranteed by law to support working parents.",
    "The use of animals for cosmetic testing should be banned to promote ethical alternatives.",
    "The use of renewable energy sources is essential in combating climate change.",
    "The government should invest more in public transportation to reduce traffic congestion and emissions.",
    "College education should be more affordable and accessible to all individuals.",
    "The portrayal of violence in media contributes to increased aggression and desensitization in society.",
    "Internet access should be considered a basic human right to bridge the digital divide.",
    "The government should implement stricter regulations on the use of pesticides in agriculture.",
    "Foreign language education should be a mandatory part of the curriculum in schools.",
    "Animal rights should be legally protected to prevent cruelty and exploitation.",
    "Nuclear disarmament is essential for global peace and security.",
    "Social media platforms should be held accountable for protecting user privacy.",
    "Genetically modified crops have the potential to solve food scarcity and improve nutrition.",
    "The use of solitary confinement in prisons constitutes cruel and inhumane punishment.",
    "Traditional gender roles are outdated and should be dismantled for gender equality.",
    "The government should provide more funding for scientific research and innovation.",
    "Internet censorship is a violation of freedom of speech and expression.",
    "Public schools should include comprehensive sex education to promote healthy relationships.",
    "The use of surveillance cameras in public places infringes upon civil liberties.",
    "Art and cultural education should be integrated into all academic disciplines.",
    "The government should implement stricter regulations on the sale and ownership of firearms.",
    "Technology has improved accessibility for individuals with disabilities.",
    "Financial incentives should be provided to individuals and businesses for adopting renewable energy.",
    "Hate speech should not be protected under freedom of speech laws.",
    "Children should have limited access to violent video games and movies.",
    "Universal basic income is a viable solution to address income inequality.",
    "The government should invest more in early childhood education for long-term societal benefits.",
    "Economic growth should not come at the expense of environmental sustainability.",
    "Prayer should not be allowed in public schools to maintain a separation of church and state.",
    "The use of non-renewable energy sources should be phased out to combat climate change.",
    "Artificial intelligence should be regulated to prevent the concentration of power in a few hands.",
    "Parents should have the right to choose whether or not to vaccinate their children.",
    "Animal farming for meat production is inherently unethical and environmentally damaging.",
    "The government should prioritize funding for the arts to promote cultural enrichment.",
    "GMO labeling should be mandatory to provide consumers with informed choices.",
    "Universal healthcare would lead to a healthier population and reduced healthcare costs.",
    "The use of plastic packaging should be reduced to address the global waste crisis.",
    "Social media platforms should be held accountable for curbing the spread of misinformation.",
    "Public schools should include comprehensive LGBTQ+ education to foster inclusivity.",
    "Nuclear energy is a necessary transition to a low-carbon future.",
    "Parental involvement is crucial for the success of students in their education.",
    "The government should impose higher taxes on the wealthy to reduce income inequality.",
    "The legal age for voting should be lowered to 16 to promote youth engagement in politics.",
    "Organic food is healthier and more environmentally sustainable than conventionally grown food.",
    "Privacy laws should be updated to address the challenges posed by new technologies.",
    "The use of animals for entertainment purposes should be strictly regulated and monitored.",
    "The government should provide free contraception to promote reproductive rights.",
    "Cultural diversity should be celebrated and valued in all aspects of society.",
    "Public funding for the arts should be increased to support creativity and cultural expression.",
    "The use of fossil fuels should be phased out to combat climate change and reduce pollution.",
    "Educational institutions should prioritize teaching critical digital literacy skills.",
    "The government should prioritize investment in renewable energy infrastructure for job creation.",
    "Media outlets should be held accountable for unbiased and objective reporting."]

if os.path.exists("thesis_embeddings.npy"):
    thesis_embeddings = np.load("thesis_embeddings.npy")
else:
    thesis_embeddings = get_minilm_embedding_batch(argumentative_theses)
    thesis_embeddings = np.array(thesis_embeddings)
    np.save("thesis_embeddings.npy", thesis_embeddings)


def get_embeddings():
    return thesis_embeddings


def get_theses():
    return argumentative_theses


def get_argumentativeness(phrase):
    phrase_embedding = np.array(get_minilm_embedding(phrase))
    return np.mean(np.matmul(thesis_embeddings, phrase_embedding))


def get_manual():
    return [
        "All lockdown measures taken were necessary and meaningfully contributed to the reduction of the spread of the virus.",
        "The rationale behind some lockdown measures was unclear and hard to understand.",
        "Regional differences between lockdown measures often led to people skirting restrictions.",
        "During the opening-up phase, many remaining restrictions seemed arbitrary and contradictory.",
        "Some technological measures introduced to fight the pandemic made life harder for elderly people and others who are not tech-savvy.",
        "Some hygiene measures introduced during the pandemic should become a new social norm.",
        "A complete lockdown should have been introduced earlier to fight the spread of the virus."
    ]
