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

    def count_a_pairs(self, string):
        """
        Count the number of valid non-overlapping pairs of 'a' characters in a string.
        A valid pair consists of any two 'a' characters, either adjacent or separated
        by any number of characters, but once a pair is found, those positions are ignored.

        Args:
        string (str): The input string.

        Returns:
        int: The number of valid non-overlapping 'a' pairs.
        """
        # Find all positions of 'a' characters in the string
        a_positions = [i for i, char in enumerate(string) if char == 'a']
        
        # Count non-overlapping pairs
        count = 0
        i = 0
        while i < len(a_positions) - 1:
            # If we find a valid pair, we skip the next position (non-overlapping)
            count += 1
            i += 2  # Move by 2 to ensure we don't overlap the pair
        
        return count


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
#list1 = ['apppa']
#list2 = ['atta']
#levenshtein_distances = distance_calculator.levenshtein_distance(list1, list2)
#print("Levenshtein Distances:", levenshtein_distances)

# Example strings for Mesh distance
#mesh_distances = distance_calculator.mesh_distance(list1, list2)
#print("Mesh Distances:", mesh_distances)
