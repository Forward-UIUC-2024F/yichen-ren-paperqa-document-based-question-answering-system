import os

def levenshtein_distance_words(s1, s2):
    words1 = s1.split()
    words2 = s2.split()
    len1 = len(words1)
    len2 = len(words2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if words1[i - 1] == words2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    
    return dp[len1][len2]

def calculate_distances(dir1, dir2, output_file):
    filenames1 = os.listdir(dir1)
    filenames2 = os.listdir(dir2)
    
    with open(output_file, 'w') as result_file:
        for filename in filenames1:
            if filename in filenames2:
                with open(os.path.join(dir1, filename), 'r') as file1, \
                     open(os.path.join(dir2, filename), 'r') as file2:
                    text1 = file1.read()
                    text2 = file2.read()
                    
                    distance = levenshtein_distance_words(text1, text2)
                    result_file.write(f"{filename}, {distance}\n")
                    print(f"{filename}, {distance}")

# Example usage:
calculate_distances('squad_answer', 'squad_answer_Llama3', 'result.txt')