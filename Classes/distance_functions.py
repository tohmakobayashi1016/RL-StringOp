class DistanceFunctions:
    def __init__(self):
        pass

    def levenshtein_distance(self, list1, list2):
        """
        Calculate the pairwise Levenshtein distance between two lists of strings.
        It assumes the lists have the same length and returns the Levenshtein distance
        for each corresponding pair of strings in the lists separately.

        Args:
        list1 (list): A list of strings.
        list2 (list): A list of strings (same length as list1).

        Returns:
        list: A list of Levenshtein distances for each pair of strings.
        """
        # Ensure both lists have the same length
        if len(list1) != len(list2):
            raise ValueError("Lists must have the same length.")
        
        distances = []
        
        # Calculate the Levenshtein distance for each pair of strings
        for str1, str2 in zip(list1, list2):
            len_str1 = len(str1)
            len_str2 = len(str2)
            
            # Create a distance matrix
            dp = [[0 for j in range(len_str2 + 1)] for i in range(len_str1 + 1)]
            
            # Initialize the matrix
            for i in range(len_str1 + 1):
                dp[i][0] = i
            for j in range(len_str2 + 1):
                dp[0][j] = j
            
            # Compute the Levenshtein distance
            for i in range(1, len_str1 + 1):
                for j in range(1, len_str2 + 1):
                    if str1[i - 1] == str2[j - 1]:
                        cost = 0
                    else:
                        cost = 1
                    dp[i][j] = min(dp[i - 1][j] + 1,    # Deletion
                                dp[i][j - 1] + 1,    # Insertion
                                dp[i - 1][j - 1] + cost)  # Substitution
            
            # Append the distance for this pair to the distances list
            distances.append(dp[len_str1][len_str2])
        
        return distances

    def count_a_pairs(self, string_list):
        """
       Args:
        string_list (list): List of strings.

        Returns:
        float: The total number of valid non-overlapping 'a' pairs across all strings plus 0.2
            for every character between the 'a's.
        """
        total_count = 0.0  # Initialize total count as a float

        for string in string_list:
            # Find all positions of 'a' characters in the current string
            a_positions = [i for i, char in enumerate(string) if char == 'a']

            # Initialize the count for this string
            count = 0.0
            i = 0
            while i < len(a_positions) - 1:
                # Check characters between the current 'a' and the next 'a'
                between = string[a_positions[i] + 1 : a_positions[i + 1]]

                # Check if the pair is valid (at least one character between, and not all 'p')
                if len(between) > 0 and not all(c == 'p' for c in between):
                    # Add 1 for the valid pair
                    count += 1
                    
                    # Add 0.2 for each character between (including 'p')
                    count += len(between) * 0.2
                    
                    # Move by 2 to ensure non-overlapping pairs
                    i += 2
                else:
                    # If an invalid pair is found, stop further counting
                    break

            # Add this string's count to the total
            total_count += count

        return total_count

    def mesh_distance(self, list1, list2):
        """
        Calculate the pairwise mesh distance between two lists of strings based on non-overlapping 'a' pairs.
        This distance counts the number of 'a' pairs in the target string
        that are missing from the current string.

        Args:
        list1 (list): A list of strings.
        list2 (list): A list of strings (same length as list1).

        Returns:
        list: A list of mesh distances (number of 'a' pairs needed) for each pair of strings.
        """
        # Ensure both lists have the same length
        if len(list1) != len(list2):
            raise ValueError("Lists must have the same length.")
        
        distances = []
        
        # Calculate the mesh distance for each pair of strings
        for current_string, target_string in zip(list1, list2):
            # Count the non-overlapping 'a' pairs in both the current and target strings
            current_a_pairs = self.count_a_pairs(current_string)
            target_a_pairs = self.count_a_pairs(target_string)
            
            # The mesh distance is the number of 'a' pairs needed
            # If the current string already has more or equal pairs, the distance is zero
            distance = max(0, target_a_pairs - current_a_pairs)
            distances.append(distance)
        
        return distances


# Example usage of the DistanceFunctions class
#distance_calculator = DistanceFunctions()

# Example strings for Levenshtein distance
#list1 = ['tt', 'atpta']

#print(distance_calculator.count_a_pairs(list1))


#list2 = ['a']
#list3 = ['p']
#list4 = ['t']
#list5 = ['at']
#print(distance_calculator.levenshtein_distance(list2, list5))
#print(distance_calculator.levenshtein_distance(list3, list5))
#print(distance_calculator.levenshtein_distance(list4, list5))


