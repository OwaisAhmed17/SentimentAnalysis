
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
import re

cEXT = pickle.load( open( "data/models/cEXT.p", "rb"))
cNEU = pickle.load( open( "data/models/cNEU.p", "rb"))
cAGR = pickle.load( open( "data/models/cAGR.p", "rb"))
cCON = pickle.load( open( "data/models/cCON.p", "rb"))
cOPN = pickle.load( open( "data/models/cOPN.p", "rb"))
vectorizer_31 = pickle.load( open( "data/models/vectorizer_31.p", "rb"))
vectorizer_30 = pickle.load( open( "data/models/vectorizer_30.p", "rb"))

def predict_personality(text):
    try:
        scentences = re.split("(?<=[.!?]) +", text)
        text_vector_31 = vectorizer_31.transform(scentences)
        text_vector_30 = vectorizer_30.transform(scentences)
        EXT = cEXT.predict(text_vector_31)
        NEU = cNEU.predict(text_vector_30)
        AGR = cAGR.predict(text_vector_31)
        CON = cCON.predict(text_vector_31)
        OPN = cOPN.predict(text_vector_31)
        predictions = [EXT[0], OPN[0], AGR[0], CON[0]]
    
        outstr = {'pers':"",'cog':"""""",'dom':"""""",'aux':"""""",'ter':"""""",'inf':"""""",'car':"""""",'int':"""""",'fri':"""""",'par':"""""",'rel':""""""}
    #0000
        if EXT[0]==0 and OPN[0]==0 and AGR[0]==0 and CON[0]==0:
            outstr['pers']="ISTP: The Crafter (Introverted, Sensing, Thinking, Perceiving)"
            outstr['cog']="""The MBTI suggests that people possess a number of different cognitive functions (thinking, sensing, feeling, and intuition) that can then be directed inwards (introverted) or outwards (extraverted). The hierarchical arrangement of these functions is what makes up each individual's personality, the MBTI suggests.

            The dominant function is the most prominent aspect of personality, although the auxiliary function also plays an important supporting role. The tertiary and inferior functions are less important and may operate on a largely unconscious basis or may become more apparent during times when a person is under stress.
            """
            outstr['dom']="Introverted Thinking: ISTPs spend a great deal of time thinking and dealing with information in their own heads. This means they do not spend a great deal of time expressing themselves verbally, so they are often known as being quiet."
            outstr['aux']="""Extraverted Sensing: ISTPs prefer to focus on the present and take on things one day at a time. They often avoid making long-term commitments and would rather focus on the "here and now" rather than think about future plans and possibilities."""
            outstr['ter']="""Introverted Intuition: This function often operates largely unconsciously in the ISTP personality. While they are not usually interested in abstract ideas, they may take such concepts and try to turn them into action or practical solutions."""
            outstr['inf']="""Extraverted Feeling: This aspect of personality tends to operate in the background of the ISTP personality, but it can become more apparent during times of stress."""
            outstr['car']="""Because ISTPs are introverted, they often do well in jobs that require working alone. """
            outstr['int']="""Depending on the type of relation you have with this person you have to interact with them differently in order to best help them through trying times."""
            outstr['fri']="""ISTPs tend to be curious and even adventurous, but they also have a strong need to be alone at times. You can be a great friend by asking them to get out and pursue new things, but be ready to respect their need for peace and quiet when they are not feeling up to going out."""
            outstr['par']="""If you are a parent to an ISTP child, you are probably well aware of their independent, adventurous nature. You can encourage their confidence by providing safe and healthy opportunities for them to explore things on their own. Provide rules and guidance, but be careful not to hover. Give your child plenty of hands-on learning, outdoor adventures, and opportunities to experiment with how things work."""
            outstr['rel']="""Because ISTPs live so strongly in the present moment, long-term commitments can be a real challenge. You can strengthen your relationship with your ISTP partner by being willing to take things day to day and by respecting their fierce need for independence."""
        #0001    
        elif EXT[0]==0 and OPN[0]==0 and AGR[0]==0 and CON[0]==1:
            outstr['pers']="""ISTJ: The Inspector (Introverted, Sensing, Thinking, Judging)"""
            outstr['cog']="""The MBTI suggests that the four different cognitive functions (thinking, feeling, intuition, and sensing) form a hierarchy. Each function is either directed outwardly (extraverted) or inwardly (introverted) and the order of these functions determines an individual's personality."""
            outstr['dom']="""Introverted Sensing: Introverted sensors are focused on the present moment, taking in an abundance of information about their surroundings."""
            outstr['aux']="""Extraverted Thinking: ISTJs are logical and efficient. They enjoy looking for rational explanations for events."""
            outstr['ter']="""Introverted Feeling: As they make judgments, ISTJs often make personal interpretations based on their internal set of values."""
            outstr['inf']="""Extraverted Intuition: This aspect of personality enjoys new ideas and experiences."""
            outstr['car']="""Because of this need for order, they tend to do better in learning and work environments that have clearly defined schedules, clear-cut assignments and a strong focus on the task at hand. """
            outstr['int']="""Depending on the type of relation you have with this person you have to interact with them differently in order to best help them through trying times."""
            outstr['fri']="""ISTJs tend to get along best with friends who are similar to themselves. While they tend to be a bit serious and by the book, they do like to have fun. They might not be willing to jump into new things, but you can be a great friend by helping them pursue hobbies and activities that they enjoy."""
            outstr['par']="""ISTJ parents tend to be quite focused on tradition and are good at providing security and stability to their children. Children of ISTJ parents often find that their parents will treat them with care and respect and that they also expect the same treatment in return."""
            outstr['rel']="""While ISTJs may experience deep feelings, they often struggle to show that side of themselves in romantic relationships. You can be an understanding partner by not expecting them to bare their soul to you right off the bat. Sometimes it may seem that your partner is not considering your feelings, but you can help them see your side by rationally presenting facts and logical explanations for your side of the argument."""
        #0010
        elif EXT[0]==0 and OPN[0]==0 and AGR[0]==1 and CON[0]==0:
            outstr['pers']="""ISFP: The Artist (Introverted, Sensing, Feeling, Perceiving)"""
            outstr['cog']="""The MBTI identifies four key cognitive functions (thinking, feeling, intuition, and sensing) that are either directed outwardly (extraverted) or inwardly (introverted)."""
            outstr['dom']="""Introverted Feeling: ISFPs care more about personal concerns rather than objective, logical information."""
            outstr['aux']="""Extraverted Sensing: People with ISFP personalities are very in tune with the world around them. They are very much attuned to sensory information and are keenly aware when even small changes take place in their immediate environment."""
            outstr['ter']="""Introverted Intuition: This function tends to run in the background, feeding off of the extraverted sensing function."""
            outstr['inf']="""Extraverted Thinking: One weakness that ISFPs may have is in organizing, although they may use this function more prominently in certain situations."""
            outstr['car']="""People with ISFP personalities love animals and have a strong appreciation for nature. They may seek out jobs or hobbies that put them in contact with the outdoors and with animals."""
            outstr['int']="""Depending on the type of relation you have with this person you have to interact with them differently in order to best help them through trying times."""
            outstr['fri']="""ISFPs are friendly and get along well with other people, but they typically need to get to know you well before they really open up."""
            outstr['par']="""ISFP tend to be perfectionists and can be their own harshest critics."""
            outstr['rel']="""ISFPs are very considerate in relationships, often to the point that they will continually defer to their partner."""
        #0011
        elif EXT[0]==0 and OPN[0]==0 and AGR[0]==1 and CON[0]==1:
            outstr['pers']="""ISFJ: The Protector (Introverted, Sensing, Feeling, Judging)"""
            outstr['cog']="""The ISFJ type relies on four key cognitive functions when taking in information and making decisions. The dominant function is the primary aspect of personality, while the auxiliary function plays a secondary and supportive role."""
            outstr['dom']="""Introverted Sensing: This function leads the introverted sensing types to focus on details and facts. ISFJs prefer concrete information rather than abstract theories. They are highly attuned to the immediate environment and firmly grounded in reality."""
            outstr['aux']="""Extraverted Feeling: ISFJs place a great emphasis on personal considerations. Extraverted feelers are focused on developing social harmony and connection."""
            outstr['ter']="""Introverted Thinking: Rather than simply trying to understand a small part of something, they want to see how things fit together and how it functions as a whole."""
            outstr['inf']="""Extraverted Intuition: While ISFJs tend to be focused on the present and on concrete facts, this largely unconscious function can help balance ISFJ personality by helping the individual focus on possibilities."""
            outstr['car']="""ISFJs have a number of characteristics that make them well-suited to particular careers. Because they are so attuned to the feelings of others, jobs in mental health or the healthcare industry are a good fit."""
            outstr['int']="""Below are suggestions for relating effectively with ISFJ personalities:"""
            outstr['fri']="""If you are friends with an ISFJ, you are probably already aware that they tend to be warm and selfless. Even though they are quite social for introverts, they are not always good at sharing their own feelings. Asking them how they are doing and being willing to talk can help them to open up."""
            outstr['par']="""ISFJs are natural caregivers and are very nurturing toward their children. They are good at giving their kids structure and order, but sometimes have a difficult time enforcing discipline.
            If you are the parent of an ISFJ child, be aware of your child's need to have time alone. Also be aware that your child may be willing to give up things that are important to them in order to make other people happy.
            """
            outstr['rel']="""ISFJs are very faithful to their partners and approach relationships with an intensity of emotion and great devotion. While they have strong feelings, they are not always good at expressing them."""
        #0100
        elif EXT[0]==0 and OPN[0]==1 and AGR[0]==0 and CON[0]==0:
            outstr['pers']="""Personality: INTP: The Thinker (Introverted, Intuitive, Thinking, Perceiving)"""
            outstr['cog']="""The MBTI is based upon psychoanalyst Carl Jung's theory which suggests that personality is made up of different cognitive functions. The hierarchical order of these functions is what establishes personality and behavioral patterns."""
            outstr['dom']="""Introverted Thinking: This function focuses on how people take in information about the world. INTPs express this by trying to understand how things work. They often like to break down larger things or ideas in order to look at the individual components in order to see how things fit and function together."""
            outstr['aux']="""Extraverted Intuition: INTPs express this cognitive function by exploring what-ifs and possibilities. They utilize insight, imagination, and past experiences to form ideas."""
            outstr['ter']="""Introverted Sensing: INTPs tend to be very detail-oriented, carefully categorizing all of the many facts and experiences that they take in."""
            outstr['inf']="""INTPs tend to seek harmony in groups. While they are introverted, INTPs can be quite outgoing when they are around people with whom they are familiar and comfortable."""
            outstr['car']="""Because they enjoy theoretical and abstract concepts, INTPs often do particularly well in science-related careers. They are logical and have strong reasoning skills, but are also excellent at thinking creatively."""
            outstr['int']="""Tips for Interacting With INTPs based on relation:"""
            outstr['fri']="""Shared interests are one of the best paths to forming a friendship with an INTP. They tend to value intellect over all else and can be very slow to form friendships. While this often leads to fewer friendships, the ones that an INTP does gain tend to be very close."""
            outstr['par']="""If your child is an INTP, it is important to remember that your child may respond better to reason and logic rather than appeals to emotion. Encourage your child to develop his or her intellectual interest, but also look for situations that may help your child foster friendships."""
            outstr['rel']="""INTPs tend to live inside their minds, so they can be quite difficult to get to know. Even in romance, they often hold back until they feel that the other person has proven themselves worthy of hearing these innermost thoughts and feelings. One thing to remember is that while INTPs do enjoy romance in the context of a deeply committed relationship, they do not play games."""
        #0101
        elif EXT[0]==0 and OPN[0]==1 and AGR[0]==0 and CON[0]==1:
            outstr['pers']="""INTJ: The Architect, (Introverted, Intuitive, Thinking, Judging)"""
            outstr['cog']="""The MBTI identifies preferences in four key dimensions: 1) Extraversion vs Introversion, 2) Sensing vs Intuition, 3) Thinking vs Feeling and 4) Judging vs Perceiving. As you can tell by the four-letter acronym, INTJ stands for Introverted, Intuitive, Thinking, and Judging."""
            outstr['dom']="""Introverted Intuition: INTJs use introverted intuition to look at patterns, meanings, and possibilities. Rather than simply looking at the concrete facts, they are more interested in what these facts mean."""
            outstr['aux']="""Extraverted Thinking: As a secondary function in the INTJ personality, extroverted thinking leads people to seek order, control, and structure in the world around them."""
            outstr['ter']="""Introverted Feeling: INTJs use introverted feeling but because it is a tertiary function, they do so to a lesser degree than they use the dominant and auxiliary functions."""
            outstr['inf']="""Extraverted Sensing: In INTJs, this tends to be the least developed of their cognitive functions, but it does still exert some influence."""
            outstr['car']="""When INTJs develop an interest in something, they strive to become as knowledgeable and skilled as they can in that area. They have high expectations, and they hold themselves to the highest possible standards."""
            outstr['int']="""Tips for Interacting With INTJs based on your relation:"""
            outstr['fri']="""INTJs tend to be solitary and self-sufficient, so establishing friendships can sometimes be difficult. Because people with this personality type tend to think so much of the future, they may avoid getting to know people because they believe that a long-term friendship will not work out."""
            outstr['par']="""INTJ parents tend to be thoughtful and attentive, yet they are typically not highly affectionate. They have high expectations for their kids and offer support by helping kids think logically when faced with decisions."""
            outstr['rel']="""Because INTJs can be difficult to get to know, romantic relationships can sometimes falter. If your partner has this personality type, it is important to know that loyalty and understanding are important. """
        #0110
        elif EXT[0]==0 and OPN[0]==1 and AGR[0]==1 and CON[0]==0:
            outstr['pers']="""INFP: The Mediator (Introverted, Intuitive, Feeling, Perceiving)"""
            outstr['cog']="""In the MBTI, each personality type is made up of a hierarchical stack of these functions. The dominant function is one that largely controls personality, although it is also supported by the auxiliary and, to a lesser degree, the tertiary functions. Inferior functions are those that are largely unconscious but still exert some influence."""
            outstr['dom']="""Introverted Feeling: INFPs experience a great depth of feelings, but as introverts they largely process these emotions internally. They possess an incredible sense of wonder about the world and feel great compassion and empathy for others. """
            outstr['aux']="""Extraverted Intuition: INFPs explore situations using imagination and 'what if' scenarios, often thinking through a variety of possibilities before settling on a course of action. Their inner lives are a dominant force in personality and they engage with the outside world by using their intuition. They focus on the "big picture" and things will shape the course of the future. This ability helps make INFPs transformative leaders who are excited about making positive changes in the world."""
            outstr['ter']="""Introverted Sensing: When taking in information, INFPs create vivid memories of the events. They will often replay these events in their minds to analyze experiences in less stressful settings. Such memories are usually associated with strong emotions, so recalling a memory can often seem like reliving the experience itself."""
            outstr['inf']="""Extraverted Thinking: This cognitive function involves organizing and making sense of the world in an objective and logical manner. While this is a largely unconscious influence in the INFPs personality, it can show itself in times of pressure. When faced with stress, an INFP might become suddenly very pragmatic and detail-oriented, focusing on logic rather than emotion. """
            outstr['car']="""INFPs typically do well in careers where they can express their creativity and vision. While they work well with others, they generally prefer to work alone. """
            outstr['int']="""Tips for Interacting With INFPs based on relation:"""
            outstr['fri']="""INFPs typically only have a few close friendships, but these relationships tend to be long-lasting. While people with this type of personality are adept at understanding others emotions, they often struggle to share their own feelings with others. Social contact can be difficult, although INFPs crave emotional intimacy and deep relationships. Getting to know an INFP can take time and work, but the rewards can be great for those who have the patience and understanding."""
            outstr['par']="""INFP parents are usually supportive, caring, and warm. They are good at establishing guidelines and helping children develop strong values. Their goal as parents is to help their children grow as individuals and fully appreciate the wonders of the world. They may struggle to share their own emotions with their children and are often focused on creating harmony in the home."""
            outstr['rel']="""As with friendships, INFPs may struggle to become close to potential romantic partners. Once they do form a relationship, they approach it with a strong sense of loyalty."""
        #0111
        elif EXT[0]==0 and OPN[0]==1 and AGR[0]==1 and CON[0]==1:
            outstr['pers']="""INFJ: The Advocate (Introverted, Intuitive, Feeling, Judging)"""
            outstr['cog']="""MBTI advocates often utilize what they refer to as a functional stack when analyzing results. You can think of the different cognitive functions as the ingredients that go into making up a personality type. The specific recipe for each type is controlled by how these different ingredients combine and interact."""
            outstr['dom']="""Introverted Intuition: This means that they tend to be highly focused on their internal insights."""
            outstr['aux']="""Extraverted Feeling: This characteristic of this type makes INFJs highly aware of what other people are feeling, but it means they are sometimes less aware of their own emotions."""
            outstr['ter']="""Introverted Thinking: INFJs rely primarily on their introverted intuition and extroverted feeling when making decisions, particularly when they are around other people. When they are alone, however, people with this personality type may rely more on their introverted thinking."""
            outstr['inf']="""Introverted Thinking: INFJs rely primarily on their introverted intuition and extroverted feeling when making decisions, particularly when they are around other people. When they are alone, however, people with this personality type may rely more on their introverted thinking."""
            outstr['car']="""INFJs do well in careers where they can express their creativity. Because people with INFJ personality have such deeply held convictions and values, they do particularly well in jobs that support these principles. INFJs often do best in careers that mix their need for creativity with their desire to make meaningful changes in the world."""
            outstr['int']="""Tips for Interacting With INFJs based on relations:"""
            outstr['fri']="""Because they are reserved and private, INFJs can be difficult to get to know. They place a high value on close, deep relationships and can be hurt easily, although they often hide these feelings from others. Interacting with an INFJ involves understanding and supporting their need to retreat and recharge."""
            outstr['par']="""Because INFJs are so skilled at understanding feelings, they tend to be very close and connected to their children. They have high standards, and can have very high behavioral expectations. They are concerned with raising children that are kind, caring, and compassionate. """
            outstr['rel']="""INFJs have an innate ability to understand other people's feelings and enjoy being in close, intimate relationships. They tend to flourish best in romantic relationships with people who they share their core values with. As a partner, it is important to provide the support and emotional intimacy that an INFJ craves."""
        #1000
        elif EXT[0]==1 and OPN[0]==0 and AGR[0]==0 and CON[0]==0:
            outstr['pers']="""ESTP: The Persuader (Extraverted, Sensing, Thinking, Perceiving)"""
            outstr['cog']="""The MBTI suggests that personality is composed of a number of different mental functions (sensing, thinking, intuition, and feeling) that are either directed inwardly (introverted) or outwardly (extraverted). """
            outstr['dom']="""Extraverted Sensing: Extraverts gain energy from social engagement. Because of this, they tend to be outgoing, engaging, and novelty-seeking. Because they are more outward-turning, they also tend to seek stimulation through the senses."""
            outstr['aux']="""Introverted Thinking: When making judgments about the world, ESTPs focus inwardly and process information in a logical and rational way. Because this side of personality is introverted, it is something that people may not immediately notice."""
            outstr['ter']="""Extraverted Feeling: This function focuses on creating social harmony and relationships with others. While emotions are not an ESTPs strongest suit, they do have a great need for social engagement."""
            outstr['inf']="""Introverted Intuition: This aspect of personality focuses on looking at information in order to see patterns and develop a "gut feeling" about situations. It allows ESTPs to gain impressions of incoming data and develop a sense of the future."""
            outstr['car']="""The MBTI also suggests that certain personality types may exhibit preferences and strengths that align them with certain careers.3 People with an ESTP personality type feel energized when they interact with a wide variety of people, so they do best in jobs that involve working with others. They strongly dislike routine and monotony, so fast-paced jobs are ideal."""
            outstr['int']="""If your friend, co-worker, loved one, or partner is an ESTP, there are some things you can do to improve your communication and interaction with them. Learning more about what makes this personality type tick can help you understand them better."""
            outstr['fri']="""ESTPs have an inexhaustible thirst for adventure. You can be a good friend by always being ready to head out for a new experience, or even by coming up with plans that offer excitement, novelty, and challenge."""
            outstr['par']="""ESTP children can be adventurous and independent, which is why parents need to set boundaries and ensure that fair, consistent discipline is used. Kids with this type of personality need lots of hands-on activities to keep them busy, but they may struggle in classroom settings where they quickly grow weary of routines."""
            outstr['rel']="""ESTPs are exciting and fun-loving, but they can grow bored with routines quickly. They do not enjoy long, philosophical discussions but like to keep the conversation flowing as they talk about shared interests and passions. Be aware that your partner prefers to take things day by day, may struggle with making long-term commitments, and has a hard time making plans for the future."""
        #1001
        elif EXT[0]==1 and OPN[0]==0 and AGR[0]==0 and CON[0]==1:
            outstr['pers']="""ESTJ: The Director (Extraverted, Sensing, Thinking, Judging)"""
            outstr['cog']="""The MBTI suggests that each personality type is made up of a number of cognitive functions (sensing, thinking, feeling, and intuition) that are either directed toward the outside world (extraverted) or inward (introverted)."""
            outstr['dom']="""Extraverted Thinking: ESTJs rely on objective information and logic to make decisions rather than personal feelings. They are skilled at making objective, impersonal, and impartial decisions. Rather than focusing on their own subjective feelings when they are making judgments, they consider facts and logic in order to make rational choices."""
            outstr['aux']="""Introverted Sensing: They are good at remembering things with a great deal of detail. Their memories of past events can be quite vivid, and they often utilize their recollections of past experiences to make connections with present events."""
            outstr['ter']="""Extraverted Intuition: This aspect of personality seeks out novel ideas and possibilities. It compels people with this personality type to explore their creativity."""
            outstr['inf']="""Introverted Feeling: When this function is used, it may lead ESTJs to make decisions based more on feelings than on logic. These are often internal valuations that lead to "gut feelings" about some situations. While this function is not used as often, in some cases a person might allow their subjective feelings to override their objective interpretation of a situation."""
            outstr['car']="""ESTJs have a wide range of personality characteristics that help them excel at a number of different careers. Their emphasis on rules and procedures make them well-suited to supervisory and management positions, while their respect for laws, authority, and order help them excel in law enforcement roles."""
            outstr['int']="""Tips for Interacting With ESTJs based on relation:"""
            outstr['fri']="""People with this personality type are sociable and enjoy getting their friends involved in activities that they enjoy. ESTJs often value dependability over almost everything else."""
            outstr['par']="""ESTJs children tend to be very responsible and goal-directed, but be cautious to avoid placing too many expectations on your child's shoulders. They enjoy structure and routine."""
            outstr['rel']="""ESTJs are dependable and take their commitments seriously. Once they have dedicated themselves to a relationship, they will stay true to it for life. They tend to avoid emotions and feelings, which can be difficult for their partners at times. """
        #1010
        elif EXT[0]==1 and OPN[0]==0 and AGR[0]==1 and CON[0]==0:
            outstr['pers']="""ESFP: The Performer (Extraverted, Sensing, Feeling, Perceiving)"""
            outstr['cog']="""The MBTI suggests that individual personalities are marked by a number of different cognitive functions (sensing, thinking, feeling, and intuition). Some of these are more dominant than others and the hierarchical order of these functions influences how people perceive and relate to the world."""
            outstr['dom']="""Extraverted Sensing: ESFPs prefer to focus on the here-and-now rather than thinking about the distant future. They also prefer learning about concrete facts rather than theoretical ideas."""
            outstr['aux']="""Introverted Feeling: People with this personality type have an internal system of values on which they base their decisions. They are very much aware of their own emotions and are empathetic towards others. They excel at putting themselves in another person's shoes, so to speak."""
            outstr['ter']="""Extraverted Thinking: This function is focused on enforcing order on the outside world. It is centered on productivity, logic, and results."""
            outstr['inf']="""Introverted Intuition: While this is the least prominent aspect of personality, this function can help the ESFP spot patterns and make connections in things they have observed."""
            outstr['car']="""With their strong dislike for routine, ESFPs do best in careers that involve a lot of variety. Jobs that involve a great deal of socializing are also a great fit, allowing individuals with this personality type to put their considerable people skills to good use."""
            outstr['int']="""Tips for Interacting With ESFPs based on relation:"""
            outstr['fri']="""ESFPs grow weary with the same old routines and are always ready for a new adventure. In order to keep up with this personality type, you need to always be ready for new experiences - from exploring new places to meeting new people."""
            outstr['par']="""ESFP children are enthusiastic and energetic, which can be both fun and exhausting for parents. You can help by providing plenty of outlets for this boundless energy. Sports, hobbies, and outdoor adventures are all good sources of fun for ESFP kids. """
            outstr['rel']="""ESFPs tend to be honest and forthright in relationships. They don't play games and are warm and enthusiastic in romantic relationships. One thing to remember is that ESFPs dislike conflict and tend to take any critical comments quite personally."""
        #1011
        elif EXT[0]==1 and OPN[0]==0 and AGR[0]==1 and CON[0]==1:
            outstr['pers']="""ESFJ: The Caregiver (Extraverted, Sensing, Feeling, Judging)"""
            outstr['cog']="""The MBTI suggests that there are a number of cognitive functions (thinking, sensing, feeling, and intuition) that help shape each individual’s personality. The hierarchical ordering of these functions is what contributes to the makeup of each personality type."""
            outstr['dom']="""Extraverted Feeling: ESFJs tend to judge people and situations based upon their "gut feelings." They often make snap decisions as a result and are quick to share their feelings and opinions."""
            outstr['aux']="""Introverted Sensing: ESFJs are more focused on the present than on the future. They are interested in concrete, immediate details rather than abstract or theoretical information."""
            outstr['ter']="""Extraverted Intuition: This cognitive function helps ESFJs make connections and find creative solutions to problems. ESFJs are known to explore the possibilities when looking at a situation. They can often find patterns that allow them to gain insights into people and experiences."""
            outstr['inf']="""Introverted Thinking: ESFJs are organized and like to plan things out in advance. Planning helps people with this personality type feel more in control of the world around them."""
            outstr['car']="""ESFJs have a number of traits that make them ideally suited to certain careers. For example, their dependability and innate need to take care of others mean that they often do well in jobs that involve supporting and caring for people such as nursing or teaching."""
            outstr['int']="""If you know someone with an ESFJ or consul personality, there are things that you can do that can help strengthen your relationship and improve your interactions."""
            outstr['fri']="""ESFJ can be selfless to the point of overlooking their own needs to ensure that other people are happy. Being a good friend to someone with this personality type means you should express your appreciation of their giving nature, but also make sure that you reciprocate their kindness."""
            outstr['par']="""ESFJ children are responsible and enjoy spending time with their family. Your child needs regular encouragement. Show your involvement by showing enthusiasm and support for their interests and activities."""
            outstr['rel']="""In romance, ESFJs are very devoted, supportive, and loyal. They are not interested in casual relationships and are focused on developing long-term commitments. You can support your partner by showing how much you love and appreciate them. Being responsive by showing affection and returning gestures of love is important."""
        #1100
        elif EXT[0]==1 and OPN[0]==1 and AGR[0]==0 and CON[0]==0:
            outstr['pers']="""ENTP: The Debater (Extroverted, Intuitive, Thinking, Perceiving)"""
            outstr['cog']="""Based upon Carl Jung's theory of personality, the MBTI categorized personality types by their cognitive functions (intuition, thinking, sensing, and feeling) which can then be directed outwardly (extroverted) or inwardly (introverted)."""
            outstr['dom']="""Extroverted Intuition: ENTPs tend to take in information quickly and are very open-minded.
            Once they have gathered this information, they spend time making connections between various complex and interwoven relationships.
            """
            outstr['aux']="""Introverted Thinking: This cognitive function is expressed in the ENTPs thinking process. People with this type of personality are more focused on taking in information about the world around them. When they do use this information to reach conclusions, they tend to be very logical."""
            outstr['ter']="""Extroverted Feeling: As a tertiary function, this aspect of personality may not be as well-developed or prominent. When developed, ENTPs can be social charmers who are able to get along well with others."""
            outstr['inf']="""Introverted Sensing: The introverted sensing function is centered on understanding the past and often applying it to current experiences and future concerns"""
            outstr['car']="""Routines and boredom are not good for the ENTP personality. They are non-conformists and do best in jobs when they can find excitement and express their creative freedom. ENTPs can be successful in a wide range of careers, as long as they do not feel hemmed in or bored."""
            outstr['int']="""Tips for Interacting With ENTPs based on relations:"""
            outstr['fri']="""ENTPs are great at getting along with people no matter what their personality type. While they are usually laid-back, they can be quite competitive. If you are friends with an ENTP, be careful not to get into the habit of trying to out-do each other."""
            outstr['par']="""ENTPs have a fun-loving nature and are excited to share their sense of wonder with their children. Parents with this personality type are supportive, but they have a tendency to try to turn every situation into a learning opportunity."""
            outstr['rel']="""In intimate relationships, ENTPs can be passionate and exciting. They are warm, loving, and good at understanding their partner's needs. You may find that they may struggle to follow through on promises that they have made, which can be a source of frustration at times."""
        #1101
        elif EXT[0]==1 and OPN[0]==1 and AGR[0]==0 and CON[0]==1:
            outstr['pers']="""ENTJ: The Commander (Extraverted, Intuitive, Thinking, Judging)"""
            outstr['cog']="""Based upon the Jungian personality theory, the MBTI suggests that personality is composed of a number of different cognitive functions. These functions can be focused primarily outward (extraverted) or inward (introverted). Each of these functions relate to how people perceive the world and make decisions."""
            outstr['dom']="""Extraverted Thinking: While they tend to make snap judgments, they are also very rational and objective. They are focused on imposing order and standards on the world around them. Setting measurable goals is important."""
            outstr['aux']="""Introverted Intuition: ENTJs are forward-thinking and are not afraid of change. They trust their instincts, although they may have a tendency to regret jumping to conclusions so quickly."""
            outstr['ter']="""Extraverted Sensing: This cognitive function gives ENTJs an appetite for adventure. They enjoy novel experiences and may sometimes engage in thrill-seeking behaviors."""
            outstr['inf']="""Introverted Feeling: Introverted feeling is centered on internal feelings and values. Emotions can be a difficult area for ENTJs, and they often lack an understanding of how this part of their personality contributes to their decision-making process."""
            outstr['car']="""Thanks to their comfort in the spotlight, ability to communicate, and a tendency to make quick decisions, ENTJs tend to naturally fall into leadership roles."""
            outstr['int']="""Tips for Interacting With ENTJs based on relation:"""
            outstr['fri']="""ENTJ are social people and love engaging conversations. While they can seem argumentative and confrontational at times, just remember that this is part of their communication style. Try not to take it personally."""
            outstr['par']="""Parents of ENTJ children should recognize that their child is independent and intellectually curious. You can help your child by allowing them to pursue their curiosity. Understand that your child will often need your reasoning explained in order to understand why certain rules need to be followed."""
            outstr['rel']="""An ENTJ partner can often seem quite dominating in a relationship. Because dealing with emotions does not come naturally to them, they may seem insensitive to their partner's feelings. It is important to remember that this does not mean that ENTJ’s don’t have feelings — they do need to feel completely comfortable in order to show their emotions."""
        #1110
        elif EXT[0]==1 and OPN[0]==1 and AGR[0]==1 and CON[0]==0:
            outstr['pers']="""ENFP: The Champion (Extraverted, Intuitive, Feeling, Perceiving)"""
            outstr['cog']="""Each personality type is composed of four cognitive functions that relate to how people process information and make decisions. It is the first two functions that play the most obvious role in personality. The latter two functions also play a role in personality, although their influence may only arise in certain settings or situations."""
            outstr['dom']="""Extraverted Intuition: ENFPs generally focus on the world of possibilities. They are good at abstract thinking and prefer not to concentrate on the tiny details."""
            outstr['aux']="""Introverted Feeling: When making decisions, ENFPs place a greater value on feelings and values rather than on logic and objective criteria. They tend to follow their heart, empathize with others, and let their emotions guide their decisions."""
            outstr['ter']="""Extraverted Thinking: This cognitive function is centered on organizing information and ideas in a logical way. When looking at information, the ENFP may use this function to sort through disparate data in order to efficiently spot connections."""
            outstr['inf']="""Introverted Sensing: ENFPs express this function by comparing the things they are experiencing in the moment to past experiences. In doing so, they are often able to call to mind memories, feelings, and senses that they associate with those events."""
            outstr['car']="""When choosing a career path, it is a good idea for people to understand the potential strengths and weaknesses of their personality type. People with the ENFP personality type do best in jobs that offer a lot of flexibility."""
            outstr['int']="""Tips for Interacting With ENFPs based on relation:"""
            outstr['fri']="""ENFPs make fun and exciting friends. They enjoy doing new things and usually have a wide circle of friends and acquaintances. They are perceptive of other people's feelings and are good at understanding other people quite quickly."""
            outstr['par']="""Because ENFPs dislike routine, their children may sometimes perceive them as inconsistent. However, they typically have strong, loving relationships with their kids and are good at imparting their sense of values. Parents of ENFP children will find that their child has a strong sense of imagination and a great deal of enthusiasm for life."""
            outstr['rel']="""ENFPs tend to be passionate and enthusiastic in romantic relationships. Long-term relationships can sometimes hit a snag because people with this personality type are always thinking about what is possible rather than simply focusing on things as they are."""
        #1111
        elif EXT[0]==1 and OPN[0]==1 and AGR[0]==1 and CON[0]==1:
            outstr['pers']="""ENFJ: The Giver (Extraverted, Intuitive, Feeling, Judging)"""
            outstr['cog']="""Each Myers-Briggs personality type can be identified by a hierarchical stack of cognitive functions that represent how each person interacts with the world. These functions focus on how people take in information about the world and how they then use this information to make decisions."""
            outstr['dom']="""Extraverted Feeling: ENFJs express this cognitive function through their engaging social behavior and harmonious social relationships. They are in tune with other people's feelings, often to the point that they ignore their own needs in order to please others."""
            outstr['aux']="""Introverted Intuition: ENFJs like to think about the future rather than the present. They may often become so focused on the larger goal that they lose sight of the immediate details."""
            outstr['ter']="""Extraverted Sensing: Extraverted sensing causes ENFJs to take in the present moment, gathering concrete details and sensory information from the environment. Because of this, they will often seek out novel or interesting experiences and sensations."""
            outstr['inf']="""Introverted Thinking: ENFJs are organized and enjoy structure and careful planning. Sticking to a predictable schedule helps ENFJs feel in control of the world around them."""
            outstr['car']="""ENFJs often do best in careers where they get to help other people and spend a great deal of time interacting with others. Because of their strong communication and organizational skills, ENFJs can make great leaders and managers."""
            outstr['int']="""Tips for Interacting With ENFJs based on relations:"""
            outstr['fri']="""One of the best ways to be a good friend to an ENFJ is to accept the care and support that they naturally offer. People with this personality type enjoy helping their friends, and it is important to show that you accept and appreciate what they have to offer."""
            outstr['par']="""Children of ENFJs might find it difficult to live up to their parents' high exceptions. At times, the ENFJ parent's hands-on approach to parenting can be stifling and make it difficult for kids to explore the world on their own terms."""
            outstr['rel']="""Because ENFJs are so sensitive to the feelings of others, your happiness is critical to your partner's happiness. Remember that your partner may even put their own needs last in order to ensure that your needs are met."""
    except Exception:
        outstr = {'pers':None,'cog':None,'dom':None,'aux':None,'ter':None,'inf':None,'car':None,'int':None,'fri':None,'par':None,'rel':None}
        predictions = [2,2,2,2]
    
    return (outstr,predictions)


