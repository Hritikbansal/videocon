OBJECT = '''Your objective is to generate a contradiction sentence using a provided "input sentence" based on a specific "misalignment scenario" called "Object Misalignment." In this scenario, you should modify a key object in the "input sentence".

Please also identify the portion of the "input sentence" you've expanded and label this as the "source." Then, specify the new elements introduced in the "sentence + object misalignment" as the "target."

Your last task is to provide a "Correct Misalignment" description, clarifying how the "input sentence" is different from the "sentence + object misalignment."

Key Requirements:
- The "sentence + object misalignment" should be plausible and could theoretically occur in real life.

Guidelines:
1. The "sentence + object misalignment" should be clearly distinguishable from the "input sentence."
2. Your replacements should be creative yet reasonable.
3. Avoid changing gender, color, or race of humans in the sentence.
4. The "Correct Misalignment" should describe how the "input sentence" diverges from the "sentence + object misalignment."

Input Sentence: a smartphone and a finger pointing to the bluetooth buttons
Sentence + Object Misalignment: a smartphone and a toe pointing to the bluetooth buttons
Source: "finger"
Target: "toe"
Correct Misalignment: a finger is pointing to the bluetooth buttons, not a toe

Input Sentence: woman plays a song on the piano
Sentence + Object Misalignment: woman plays a song on the cello
Source: "piano:
Target: "cello"
Correct Misalignment: woman plays a song on the piano, not a cello

Input Sentence: a man is going in the wheel skate
Sentence + Object Misalignment: a man is going in the bicycle
Source: "wheel skate"
Target: "bicycle"
Correct Misalignment: a man is going in the wheel skate, not a bicycle 

Now it's your turn.

Input Sentence: <c>
Sentence + Object Misalignment:
Source:
Target:
Correct Misalignment:
'''

ACTION = '''Your objective is to generate a contradiction sentence using a provided "input sentence" based on a specific "misalignment scenario" called "Action Misalignment." In this scenario, you should modify specific action performed by the object in the "input sentence".

Please also identify the portion of the "input sentence" you've expanded and label this as the "source." Then, specify the new elements introduced in the "sentence + action misalignment" as the "target."

Your last task is to provide a "Correct Misalignment" description, clarifying how the "input sentence" is different from the "sentence + action misalignment."

Key Requirements:
- The "sentence + action misalignment" should be plausible and could theoretically occur in real life.

Guidelines:
1. The "sentence + action misalignment" should be clearly distinguishable from the "input sentence."
2. Your replacements should be creative yet reasonable.
3. Avoid changing gender, color, or race of humans in the sentence.
4. The "Correct Misalignment" should describe how the "input sentence" diverges from the "sentence + action misalignment."

Input Sentence: a person repairing the car
Sentence + Action Misalignment: a person driving the car
Source: "repairing"
Target: "driving"
Correct Misalignment: a person is repairing the car instead of the driving it

Input Sentence: a woman is singing
Sentence + Action Misalignment: a woman is yelling
Source: "singing"
Target: "yelling"
Correct Misalignment: a woman is singing, not yelling

Input Sentence: an animated cartoon of a monster catching a man by the foot and then launching him like a slingshot
Sentence + Action Misalignment: an animated cartoon of a monster throwing a man by the foot and then launching him like a slingshot
Source: "catching a man"
Target: "throwing a man"
Correct Misalignment: a monster is catching a man, not throwing the man

Input Sentence: a robot is entering a hall talking to a person
Sentence + Action Misalignment: a robot is leaving a hall talking to a person
Source: "entering"
Target: "leaving"
Correct Misalignment: a robot is entering a hall, not leaving it

Now it's your turn.

Input Sentence: <c>
Sentence + Action Misalignment:
Source:
Target:
Correct Misalignment:
'''

ATTRIBUTE =  '''Your objective is to generate a contradiction sentence using a provided "input sentence" based on a specific "misalignment scenario" called "Attribute Misalignment." In this scenario, you should modify an attribue of an object in the "input sentence".

Please also identify the portion of the "input sentence" you've expanded and label this as the "source." Then, specify the new elements introduced in the "sentence + attribute misalignment" as the "target."

Your last task is to provide a "Correct Misalignment" description, clarifying how the "input sentence" is different from the "sentence + attribute misalignment."

Key Requirements:
- The "sentence + attribute misalignment" should be plausible and could theoretically occur in real life.

Guidelines:
1. The "sentence + attribute misalignment" should be clearly distinguishable from the "input sentence."
2. Your replacements should be creative yet reasonable.
3. Avoid changing gender, color, or race of humans in the sentence.
4. The "Correct Misalignment" should describe how the "input sentence" diverges from the "sentence + attribute misalignment."

Input Sentence: man in blue shirt is test driving his new car
Sentence + Attribute Misalignment: man in red shirt is test driving his new car
Source: "blue"
Target: "red"
Correct Misalignment: a man in blue shirt, not in the red shirt

Input Sentence: dog with a purple hat
Sentence + Attribute Misalignment: dog with a greeen hat
Source: "purple"
Target: "green"
Correct Misalignment: dog should have a purple hat, not a green hat

Input Sentence: a group of people playing with giant beach balls
Sentence + Attribute Misalignment: a group of people playing with small beach balls
Source: "giant"
Target: "small"
Correct Misalignment: a group of people are playing with giant beach balls, not with the small beach balls

Input Sentence: there is a man with serious face looking cruelly
Sentence + Attribute Misalignment: there is a man with happy face looking kindly
Source: "serious face looking cruelly"
Target: "happy face looking kindly"
Correct Misalignment: a man is with the serious face looking cruelly, not with the happy face looking kindly

Now it's your turn.

Input Sentence: <c>
Sentence + Attribute Misalignment:
Source:
Target:
Correct Misalignment:
'''

RELATION = '''Your objective is to generate a contradiction sentence using a provided "input sentence" based on a specific "misalignment scenario" called "Relation Misalignment." In this scenario, you should change the relation between the objects in the sentence.

Please also identify the portion of the "input sentence" you've expanded and label this as the "source." Then, specify the new elements introduced in the "sentence + relation misalignment" as the "target."

Your last task is to provide a "Correct Misalignment" description, clarifying how the "input sentence" is different from the "sentence + relation misalignment."

Key Requirements:
- The "sentence + relation misalignment" should be plausible and could theoretically occur in real life.
- Relation is a word or group of words used before a noun, pronoun, or noun phrase to show direction, time, place, location, spatial relationships, or to introduce an object. Examples include: "above", "below", "inside", "outside", "front of", "behind", "up", "down", "left", "right" etc.
- Only focus on the relations between the objects; do not replace or remove any existing objects, actions or attributes in the "input sentence."

Guidelines:
1. The "target" should introduce a contradiction when compared to the "source," without being a mere negation.
2. The "sentence + relation misalignment" should be clearly distinguishable from the "input sentence."
3. Your additions should be creative yet reasonable.
4. Avoid changing gender, color, or race of humans in the sentence.
5. The "Correct Misalignment" should describe how the "input sentence" diverges from the "sentence + relation misalignment."

Input Sentence: people are dancing and singing outside
Sentence + Relation Misalignment: people are dancing and singing inside the club
Source: "outside"
Target: "inside the club"
Correct Misalignment: people are dancing and singing outside the club, not inside the club

Input Sentence: a woman talking in front of a camera
Sentence + Relation Misalignment: a woman is talking behind a camera
Source: "in front of a camera"
Target: "behind a camera"
Correct Misalignment: a woman talks in front of a camera, not behind the camera

Input Sentence: a bowl of grey shrimp is shown above a yellow broth
Sentence + Relation Misalignment: a bowl of grey shrimp is shown below a yellow broth
Source: "above"
Target: "below"
Correct Misalignment: a bowl of grey shrimp is shown above a yellow broth, not below the yellow broth

Input Sentence: a kid flips over a mattress on a trampoline
Sentence + Relation Misalignment: a kid flips over a mattress under the trampoline
Source: "on a trampoline"
Target: "under the trampoline"
Correct Misalignment: a kid flips the mattress on a trampoline, not under the trampoline

Input Sentence: a cat yawning to the left of a dog
Sentence + Relation Misalignment: a cat yawning to the right of a dog
Source: "left of a dog"
Target: "right of a dog"
Correct Misalignment: a cat is yawning to the left of a dog, not to the right of the dog

Input Sentence: a cup is placed under the table
Sentence + Relation Misalignment: a cup is placed on top of a table
Source: "under the table"
Target: "on top of a table"
Correct Misalignment: a cup is placed under the table, not on the top of the table

Input Sentence: a second hand automobile dealer salesman explains the good deal of a land rover which has done over fifty two thousand miles of running
Sentence + Relation Misalignment: a second hand automobile dealer salesman explains the good deal of a land rover which has done under fifty two thousand miles of running
Source: "over fifty two thousand miles"
Target: "under fifty two thousand miles"
Correct Misalignment: land rover has done over fifty two thousand miles of running, not of less than fifty two thousand miles

Input Sentence: a person is climbing up the tree
Sentence + Relation Misalignment: a person is climbing down the tree
Source: "climbing up"
Target: "climbing down"
Correct Misalignment: a person is climbing down the tree, not up the tree

Input Sentence: the objects are placed far away from each other
Sentence + Relation Misalignment: the objects are placed close to each other
Source: "far away"
Target: "close"
Correct Misalignment: the objects are placed far away from each other, not close to each other

Input Sentence: a live action racing video game shows a black sports car moving up the hill
Sentence + Relation Misalignment: a live action racing video game shows a black sports car moving down the hill
Source: "up the hill"
Target: "down the hill"
Correct Misalignment: sports car is moving up the hill, not down the hill

Input Sentence: people driving down the road recording themselves
Sentence + Relation Misalignment: people driving up the road recording themselves
Source: "down"
Target: "up"
Correct Misalignment: people are driving down the road, not up the road

Now it's your turn.

Input Sentence: <c>
Sentence + Relation Misalignment:
Source:
Target:
Correct Misalignment:
'''


HALLUCINATION = '''Your objective is to generate a contradiction sentence using a provided "input sentence" based on a specific "misalignment scenario" called "Hallucination Misalignment." In this scenario, you should add new elements to the sentence without replacing or removing anything that is already there.

Please also identify the portion of the "input sentence" you've expanded and label this as the "source." Then, specify the new elements introduced in the "sentence + hallucination misalignment" as the "target."

Your last task is to provide a "Correct Misalignment" description, clarifying how the "input sentence" is different from the "sentence + hallucination misalignment".

Key Requirements:
- The "sentence + hallucination misalignment" should be plausible and could theoretically occur in real life.
- Only add elements; do not replace or remove any existing elements in the "input sentence."

Guidelines:
1. The "target" should introduce a contradiction when compared to the "source," without being a mere negation.
2. The "sentence + hallucination misalignment" should be clearly distinguishable from the "input sentence."
3. Your additions should be creative yet reasonable.
4. Avoid changing gender, color, or race of humans in the sentence.
5. The "Correct Misalignment" should describe how the "input sentence" diverges from the "sentence + hallucination."

Input Sentence: A cola bottle is shown and then it is tossed
Sentence + Hallucination Misalignment: A cola bottle is shown and then it is tossed along with a frisbee
Source: "tossed"
Target: "tossed along with a frisbee"
Correct Misalignment: The cola bottle is shown and tossed but there is no frisbee being tossed along it

Input Sentence: A person is playing a video game where they become aggressive towards a woman robot face
Sentence + Hallucination Misalignment: A person is playing a video game where they become aggressive and release fireworks towards a woman robot face
Source: "aggressive towards"
Target: "aggressive and release fireworks towards"
Correct Misalignment: The person becomes aggressive towards a woman robot face but they do not release fireworks at woman robot face

Input Sentence: A man is walking his dog
Sentence + Hallucination Misalignment: A man is walking his dog while carrying a surfboard
Source: "walking his dog"
Target: "walking his dog while carrying a surfboard"
Correct Misalignment: The man is walking a dog but does not carry a surfboard

Input Sentence: Children are playing in the park
Sentence + Hallucination Misalignment: Children are playing in the park near a giant sculpture
Source: "playing in the park"
Target: "playing in the park near a giant sculpture"
Correct Misalignment: Children are playing in the park but there is no giant sculpture in the park

Input Sentence: A woman is reading a book
Sentence + Hallucination Misalignment: A woman is reading a book under a parasol
Source: "reading a book"
Target: "reading a book under a parasol"
Correct Misalignment: The woman is reading a book but there is no parasol

Remember: Only add elements; do not replace or remove any existing elements in the "input sentence". Now it's your turn.

Input Sentence: <c>
Sentence + Hallucination Misalignment:
Source:
Target:
Correct Misalignment:
'''


COUNTING = '''Your objective is to generate a contradiction sentence using a provided "input sentence" based on a specific "misalignment scenario" called "Counting Misalignment." In this scenario, you should modify the mathematical count of the objects in the "input sentence".

Please also identify the portion of the "input sentence" you've expanded and label this as the "source." Then, specify the new elements introduced in the "sentence + counting misalignment" as the "target."

Your last task is to provide a "Correct Misalignment" description, clarifying how the "input sentence" is different from the "sentence + counting misalignment."

Key Requirements:
- The "sentence + counting misalignment" should be plausible and could theoretically occur in real life.
- Only focus on the counts of the objects; do not replace or remove any existing objects, actions or attributes in the "input sentence."

Guidelines:
1. The "sentence + counting misalignment" should be clearly distinguishable from the "input sentence."
2. Avoid changing gender, color, or race of humans in the sentence.
3. The "Correct Misalignment" should describe how the "input sentence" diverges from the "sentence + counting misalignment."

Input Sentence: a man is entering a room with three surgeons
Sentence + Counting Misalignment: a man is entering a room with one surgeon
Source: "three surgeons"
Target: "one surgeon"
Correct Misalignment: the man enters the room with three surgeons, not one surgeon

Input Sentence: three girls singing on stage on the voice
Sentence + Counting Misalignment: six girls singing on stage on the voice
Source: "three girls"
Target: "six girls"
Correct Misalignment: three girls are singing on the voice, not six girls

Input Sentence: a video showcasing 6 different peoples reactions to a certain video the video seemed family oriented
Sentence + Counting Misalignment: a video showcasing 2 different peoples reactions to a certain video the video seemed family oriented
Source: "6 different peoples reactions"
Target: "4 different peoples reactions"
Correct Misalignment: six different people were showcasing their reactions to a video, not four different people

Now it's your turn.

Input Sentence: <c>
Sentence + Counting Misalignment:
Source:
Target:
Correct Misalignment:
'''

EVENT = '''Your objective is to generate a contradiction sentence using a provided "input sentence" based on a specific "misalignment scenario" called "Event Misalignment." In this scenario, you should change the temporal order of the events in the sentence.

Your last task is to provide a "Correct Misalignment" description, clarifying how the "input sentence" is different from the "sentence + event misalignment."

Key Requirements:
- The "sentence + event misalignment" should be plausible and could theoretically occur in real life.
- Only focus on the temporal order; do not replace or remove any existing objects, actions or attributes in the "input sentence."

Guidelines:
1. The "target" should introduce a contradiction when compared to the "source," without being a mere negation.
2. The "sentence + event misalignment" should be clearly distinguishable from the "input sentence."
3. Your changes should be creative yet reasonable.
4. Avoid changing gender, color, or race of humans in the sentence.
5. The "Correct Misalignment" should describe how the "input sentence" diverges from the "sentence + event misalignment."

Input Sentence: A girl pretends to sneeze and drops something out of her hands and her friend starts to laugh and drops the phone
Sentence + Event Misalignment: A girl drops something out of her hands and then pretends to sneeze and her friend starts to laugh and drops the phone
Correct Misalignment: A girl first sneezes and then drops something out of her hands, not drops something out of her hands and then sneezes

Input Sentence: A boy is throwing a ball against a wall and a girl takes the ball and throws it.
Sentence + Event Misalignment: A girl takes the ball and throws it before the boy throws the ball against a wall
Correct Misalignment: A boy is throws the ball against the wall before the girl takes it and throws it, not a girl taking the ball and throwing it before the boy throws it against a wall

Input Sentence: A small crowd watches as a competitor performs a triple jump, then walks back to the starting mark.
Sentence + Event Misalignment: A small crowd watches a competitor walk to the starting mark, then perform a triple jump
Correct Misalignment: A competitor performs the triple jump before walking back to the starting mark, not after walking back to the starting mark

Input Sentence: A man wearing a black t-shirt is holding a cup of food in his right hand. He moves around a piece of food in his left hand to play with the ostrich
Sentence + Event Misalignment: A man wearing a black t-shirt moves around a piece of food in his left hand to play with the ostrich before holding a cup of food in his right hand
Correct Misalignment: A man is holding a cup of food before he moves around a piece of food to play with the ostrich, not after he moves around to play with the ostrich

Input Sentence: A person is playing in the doorway, then they begin laughing and grab a doorknob and leave the room
Sentence + Event Misalignment: A person is playing in the doorway, then they grab a doorknob and leave the room, and then they begin laughing
Correct Misalignment: They begin laughing before they grabbed the doorknob and leave the room, not after they grabbed a doorknob and leave the room

Input Sentence: Little girl eats from cup after the child walks downhill
Sentence + Event Misalignment: The child walks downhill after the little girl eats from cup
Correct Misalignment: First, the child walks downhill, then the little girl eats from cup, not walks downhill after eating from the cup

Now it's your turn.

Input Sentence: <c>
Sentence + Event Misalignment:
Correct Misalignment:
'''