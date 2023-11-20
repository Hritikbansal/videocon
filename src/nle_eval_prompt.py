FEEDBACK_EVAL = '''Your task is decide whether the hypothesis entails the premise or not given a context. The premise will consist "original caption" of a video, "candidate caption" of the video, "ground-truth misalignment" between the "original caption" and "candidate caption" which summarizes how the original caption is different from the candidate caption. Hypothesis will be the predicted misalignment between the "original caption" and "candidate caption".  You need to output 1 if hypothesis entails the premise for the given context otherwise 0.


Premise:
Original Caption: the fountain sprays top to bottom after the video turns sideways
Candidate Caption: the fountain sprays bottom to top after the video turns sideways
Ground-truth Misalignment: ,"the fountain sprays top to bottom, not bottom to top

Hypothesis: the fountain sprays bottom to top before the video turns sideways, not after the video turns sideways

Entailment: 0


Premise:
Original Caption: a single bush on the screen then the camera pans to the zebras on the right
Candidate Caption: A single bush on the screen then the camera pans to the zebras on the left
Ground-truth Misalignment: The camera pans to the zebras on the right, not the left

Hypothesis: A single bush on the screen, then the camera pans to the zebras on the right, not on the left

Entailment: 1


Now it's your turn.

Premise:
Original Caption: <c1>
Candidate Caption: <c2>
Ground-truth Misalignment: <c3>

Hypothesis: <c4>

Entailment: '''