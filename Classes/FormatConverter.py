class FormatConverter:
    def __init__(self):
        #Define the mapping of discrete actions to letter actions 
        self.action_map = {0: 'a', 1: 't', 2: 'p', 3: 'd'}

    def from_discrete_to_letter(self, discrete_actions):
        #Convert the list of discrete actions to corresponding letter actions
        letter_actions = [self.action_map[action] for action in discrete_actions]
        return ''.join(letter_actions)
    
    def from_string_to_vector(self, string):
        vector = []
        for k in string:
            if k == 't':
                vector.append("00")
            elif k == 'p':
                vector.append("01")
            elif k == 'a':
                vector.append("10")
            elif k == 'd':
                vector.append("11")
        # Join the elements of the vector into a single string and return it as a list with one item
        return ''.join(vector)

# Create an instance of the converter
converter = FormatConverter()

# Example discrete actions
#discrete_actions = [0, 1, 2, 0, 2, 1]

# Convert discrete actions to letter string
#letter_string = converter.from_discrete_to_letter(discrete_actions)
#print("Letter String:", letter_string)

# Convert letter string to vector
#vector = converter.from_string_to_vector(letter_string)
#print("Vector:", vector)

