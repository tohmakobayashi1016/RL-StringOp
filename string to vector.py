class StringVectorConverter:
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
        return [''.join(vector)]

# Create an instance of the class
converter = StringVectorConverter()

# Call the method with a string
input_string = "tpadt"
vector_list = converter.from_string_to_vector(input_string)

# Print the result
print(f"Input string: {input_string}")
print(f"Output list: {vector_list}")
