class FormatConverter:
    def __init__(self):
        #Define the mapping of discrete actions to letter actions 
        self.action_map = {0: 'a', 1: 't', 2: 'p', 3: 'd'}

        # Define the reverse mapping from vector to letters
        self.vector_map = {"10": 'a', "00": 't', "01": 'p', "11": 'd'}

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
    
    def from_vector_to_string(self, vector):
        # Split the vector into pairs of two (e.g. '10', '00') and convert to letters
        letters = []
        for i in range(0, len(vector), 2):
            pair = vector[i:i+2]
            letters.append(self.vector_map[pair])
        # Join the letters into a string and return it
        return ''.join(letters)

    def convert_list_of_vectors_to_strings(self, vectors):
        # Iterate over a list of vector strings and convert each to a letter string
        letter_strings = []
        for vector in vectors:
            letter_string = self.from_vector_to_string(vector)
            letter_strings.append(letter_string)
        return letter_strings

# Create an instance of the converter
#converter = FormatConverter()

# Example list of vector strings
#vector_list = ['1010000100', '1100110000', '1100011101', '0111100111', '1101100000']

# Convert each vector to a letter string
#letter_strings = converter.convert_list_of_vectors_to_strings(vector_list)

# Print the result
#print(f"Letter Strings: {letter_strings}")