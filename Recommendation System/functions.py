# Importing all the necessary libraries and modules
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from scipy.spatial import distance
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Define a function to get the recommendations for a specific movie
def get_recommendations(movie,n_predictions,pivot_table):
    '''
    Gives recommendation for movies based on cosine distance and correlation

    Args:
        `movie`: Takes in the name of the movie (str) or the index of the movie in the pivot table (int).
        `n_predictions`: The number of movie recommendations the user wants.
        `pivot_table`: Pivot table of movies consisting of movie_names (index), list of users (columns) and the rating each user gives to each movie (data).
    
    Returns:
        A sorted DataFrame consisting of recommended movies with their cosine distances, correlation, and average score.
    '''

    # Get the movies with the most correlation with the given movie
    movie_corr = get_recommendations_corr(movie,n_predictions,pivot_table)
    # Get the movies with the least cosine distance with the given movie
    movie_cosine = get_recommendations_KNN(movie,n_predictions,pivot_table)
    # Create a DataFrame of movies gathered on the basis of Correlation
    correlation_dataframe = pd.DataFrame(movie_corr.values(),index=movie_corr.keys(),columns=['Correlation'])
    # Create a DataFrame of movies gathered on the basis of Cosine distances
    cosine_dataframe = pd.DataFrame(movie_cosine.values(),index=movie_cosine.keys(),columns=['Cosine_distance'])
    # Merge both the DataFrames
    recommendations = pd.merge(cosine_dataframe,correlation_dataframe,left_index=True,right_index=True)
    # Convert the index lists of both the DataFrames to set and get the non-common movies
    cosine_index, correlation_index = set(cosine_dataframe.index), set(correlation_dataframe.index)
    non_common_movies = cosine_index ^ correlation_index
    # If there are movies that are not common to Correlation DataFrame and Cosine DataFrame...
    if(non_common_movies):
        # Initialise empty list to store the Correlation and Cosine distances of non-common movies with the given movie
        corr = []
        cosine = []
        # Checking the datatype of the variable `movie` and extracting its vector from the Pivot Table
        if(type(movie) == str):
            curr_movie = pivot_table[pivot_table.index == movie].to_numpy().flatten()
        elif(type(movie) == int):
            curr_movie = pivot_table.iloc[movie,:].to_numpy().flatten()
        # Converting the set into list for iteration
        non_common_movies = list(non_common_movies)
        for mov in non_common_movies:
            # extracting the movie vector from the Pivot Table 
            mov_vector = np.squeeze(pivot_table[pivot_table.index == mov].to_numpy())
            # Calculating the pearson's Correlation value for each movie with the given movie
            f = pearsonr(curr_movie,mov_vector)[0]
            # Appending the correlation value of the movie to the list
            corr.append(f)
            # Compute the cosine distance of the movie with the given movie and append it to the list
            cosine.append(distance.cosine(curr_movie,pivot_table[pivot_table.index == mov].to_numpy().flatten()))
        # Initialise a dictionary with Correlation and Cosine distance
        new_dict = {"Correlation": corr,"Cosine_distance":cosine}
        # Convert the dictionary to a pandas DataFrame
        dataframe = pd.DataFrame(data=new_dict,index=non_common_movies)
        # Concatenate the obtained DataFrame with the merged DataFrame obtained earlier
        recommendations = pd.concat([dataframe,recommendations])
    # Initialise MinMax scalars for Correlation and Cosine distances columns
    corr_scalar = MinMaxScaler()
    cosine_scalar = MinMaxScaler()
    # Normalise the Cosine distances and Correlation Columns for an unbiased average
    recommendations['Cosine_distance'] = cosine_scalar.fit_transform(recommendations['Cosine_distance'].to_numpy().reshape(-1,1))
    recommendations['Correlation'] = corr_scalar.fit_transform(recommendations['Correlation'].to_numpy().reshape(-1,1))
    recommendations['score'] = 0.5*recommendations['Correlation'] - 0.5*recommendations['Cosine_distance']
    # Sort the DataFrames based on Score and return the DataFrame
    recommendations = recommendations.sort_values(by='score',ascending=False).head(n_predictions)
    return recommendations

# Define a Function to get the movies with the most Correlation with the given movie
def get_recommendations_corr(movie,n_predictions,pivot_table):
    '''
    Gives recommendation based on the best correlation value with other movies

    Args:
        `movie`: Takes in the name of the movie (str) or the index of the movie in the pivot table (int).
        `n_predictions`: The number of movie recommendations the user wants.
        `pivot_table`: Pivot table of movies consisting of movie_names (index), list of users (columns) and the rating each user gives to each movie (data).
    
    Returns:
        A dictionary with the name of the recommended movies as keys and their correlation values as values.
    '''
    # Creating a tranpose of pivot table with name of the movies as columns and users as rows
    pivot_table_T = pivot_table.transpose()
    # Checking the datatype of the variable `movie` and extracting its vector from the Pivot Table
    if(type(movie) == str):
        movie_vector = pivot_table_T[movie]
    elif(type(movie) == int):
        movie_vector = pivot_table_T.iloc[:,movie]
    # Extract the movie vectors of those movies who's correlation with the given movie is the highest
    answer = pivot_table_T.corrwith(movie_vector).sort_values(ascending=False).iloc[1:n_predictions+1]
    # Return a dictionary with the keys as name of the movies and values as the Correlation value with the given movie
    return dict(zip(list(answer.index), list(answer)))

def get_recommendations_KNN(movie,n_predictions,pivot_table):
    '''
    Gives recommendation based on the least Cosine distance with other movies

    Args:
        `movie`: Takes in the name of the movie (str) or the index of the movie in the pivot table (int).
        `n_predictions`: The number of movie recommendations the user wants.
        `pivot_table`: Pivot table of movies consisting of movie_names (index), list of users (columns) and the rating each user gives to each movie (data).
    
    Returns:
        A dictionary with the name of the recommended movies as keys and their correlation values as values.
    '''
    # Initialise a NearestNeighbour model and fit it to the Pivot Table
    model = NearestNeighbors(n_neighbors=5, algorithm='auto',metric='cosine')
    model.fit(pivot_table)
    # Checking the datatype of the variable `movie` and extracting its vector from the Pivot Table
    if(type(movie) == str):
        movie_vector = pivot_table[pivot_table.index == movie].to_numpy()
    elif(type(movie) == int):
        movie_vector = pivot_table.iloc[movie,:].to_numpy()
    # Get the distance and indices of the movies with the least Cosine distance from the given movie
    distances,answers = model.kneighbors(movie_vector,n_predictions+1)
    # Flatten the obtained arrays for further processing
    distances,answers = distances.flatten(),answers.flatten()
    # Use list comprehension to make the answers array contain the name of the movies instead of their indices in the Pivot Table
    answers = [pivot_table.index[i] for i in answers]
    answers = list(answers)
    # Convert both the arrays into a dictionary with keys as the name of the movies and values as the distances of those movies from the given movie
    ans = dict(zip(answers[1:],distances[1:]))
    return ans