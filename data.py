'''data.py
Reads CSV files, stores data, access/filter data by variable name
Ahmed Rafique Choudhry
CS 251/2: Data Analysis and Visualization
Fall 2024
'''
import numpy as np

class Data:
    '''Represents data read in from .csv files
    '''
    def __init__(self, filepath=None, headers=None, data=None, header2col=None, cats2levels=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0
        cats2levels: Python dictionary or None.
                Maps each categorical variable header (var str name) to a list of the unique levels (strings)
                Example:

                For a CSV file that looks like:

                letter,number,greeting
                categorical,categorical,categorical
                a,1,hi
                b,2,hi
                c,2,hi

                cats2levels looks like (key -> value):
                'letter' -> ['a', 'b', 'c']
                'number' -> ['1', '2']
                'greeting' -> ['hi']

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - cats2levels
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col
        self.cats2levels = cats2levels
        if filepath != None:
            self.read(filepath)

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called `self.data` at the end
        (think of this as a 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if there should be nothing returned

        TODO:
        1. Set or update your `filepath` instance variable based on the parameter value.
        2. Open and read in the .csv file `filepath` to set `self.data`.
        Parse the file to ONLY store numeric and categorical columns of data in a 2D tabular format (ignore all other
        potential variable types).
            - Numeric data: Store all values as floats.
            - Categorical data: Store values as ints in your list of lists (self.data). Maintain the mapping between the
            int-based and string-based coding of categorical levels in the self.cats2levels dictionary.
        All numeric and categorical values should be added to the SAME list of lists (self.data).
        3. Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        4. Be sure to set the fields: `self.headers`, `self.data`, `self.header2col`, `self.cats2levels`.
        5. Add support for missing data. This arises with there is no entry in a CSV file between adjacent commas.
            For example:
                    letter,number,greeting
                    categorical,categorical,categorical
                     a,1,hi
                     b,,hi
                     c,,hi
            contains two missing values, in the 4th and 5th rows of the 2nd column.
            Handle this differently depending on whether the missing value belongs to a numeric or categorical variable.
            In both cases, you should subsitute a single constant value for the current value to your list of lists (self.data):
            - Numeric data: Subsitute np.nan for the missing value.
            (nan stands for "not a number" — this is a special constant value provided by Numpy).
            - Categorical data: Add a categorical level called 'Missing' to the list of levels in self.cats2levels
            associated with the current categorical variable that has the missing value. Now proceed as if the level
            'Missing' actually appeared in the CSV file and make the current entry in your data list of lists (self.data)
            the INT representing the index (position) of 'Missing' in the level list.
            For example, in the above CSV file example, self.data should look like:
                [[0, 0, 0],
                 [1, 1, 0],
                 [2, 1, 0]]
            and self.cats2levels would look like:
                self.cats2levels['letter'] -> ['a', 'b', 'c']
                self.cats2levels['number'] -> ['1', 'Missing']
                self.cats2levels['greeting'] -> ['hi']

        NOTE:
        - In any CS251 project, you are welcome to create as many helper methods as you'd like. The crucial thing is to
        make sure that the provided method signatures work as advertised.
        - You should only use the basic Python to do your parsing. (i.e. no Numpy or other imports).
        Points will be taken off otherwise.
        - Have one of the CSV files provided on the project website open in a text editor as you code and debug.
        - Run the provided test scripts regularly to see desired outputs and to check your code.
        - It might be helpful to implement support for only numeric data first, test it, then add support for categorical
        variable types afterward.
        - Make use of code from Lab 1a!
        '''
        
        #Opening and reading the file.
        self.filepath = filepath
        csv_file = open(self.filepath, "r")

        lines = csv_file.readlines()

        all_headers = lines[0].strip().split(",")
        type_line = lines[1].strip().split(",")

        #Initializing required instance variables.
        self.headers = []
        self.data = []
        
        numeric_list = []
        categoric_list = []

        for i in range(len(type_line)):
            if type_line[i].strip() == "numeric":
                numeric_list.append(i)
                self.headers.append(all_headers[i].strip())
            elif type_line[i].strip() == "categorical":
                categoric_list.append(i)
                self.headers.append(all_headers[i].strip())

        self.header2col = {}
        for i in range(len(self.headers)):
            self.header2col[self.headers[i]] = i
        
        self.cats2levels = {}
        for i in range(len(all_headers)):
            if i in categoric_list:
                self.cats2levels[all_headers[i]] = []
        
        #Indexing the lines to only read the data values and parsing them into the self.data array.
        lines = lines[2:]
        for line in lines:
            new_line = line.strip().split(",")
            new_list = []
            for j in range(len(new_line)):
                if j in numeric_list:
                    if new_line[j] == "":
                        new_list.append(np.nan)
                    else:
                        new_list.append(float(new_line[j]))
                elif j in categoric_list:
                    if new_line[j] == "":
                        new_list.append("Missing")
                    else:
                        new_list.append(new_line[j].strip())
                if not numeric_list and not categoric_list:
                    print("Error: The second row with the variable types is missing.")
                    exit()

            new_line = new_list

            for item in self.cats2levels.keys():
                value = self.header2col[item]
                if new_line[value] not in self.cats2levels[item]:
                    self.cats2levels[item].append(new_line[value])
                new_line[value] = self.cats2levels[item].index(new_line[value])

        
            self.data.append(new_line)

        self.data = np.array(self.data)
        csv_file.close()

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's called to determine
        what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.

        NOTE: It is fine to print out int-coded categorical variables (no extra work compared to printing out numeric data).
        Printing out the categorical variables with string levels would be a small extension.
        '''
        #Creating the toString using the given format in the test code
        data_string = "\n" +" -------------------------------"

        if self.data is None:
            data_string += "No data available"
            return data_string
        
        #Adding detailed information about the csv file
        data_string += "\n" + self.filepath + " " + "(" + str(self.data.shape[0]) + 'x' + str(self.data.shape[1]) + ")" + "\n" + "Headers: " + "\n"
        for header in self.headers:
            data_string += "  " + header + "  "
        
        data_string += "\n"

        #Choosing the minimum of 5 and the number of actual rows of the file. This is handy if there are less than 5 rows in the file.
        rows = min(5, self.data.shape[0])

        data_string += "Showing first " + str(rows) + "/" + str(self.data.shape[0]) + " rows." + "\n"
        for row in self.data[:rows]:
            for value in row:
                data_string += str(value) + "     "
            data_string += "\n"

        return data_string
    
    def get_headers(self):
        '''Get list of header names (all variables)

        Returns:
        -----------
        Python list of str.
        '''
        return self.headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col

    def get_cat_level_mappings(self):
        '''Get method for mapping between categorical variable names and a list of the respective unique level strings.

        Returns:
        -----------
        Python dictionary. str -> list of str
        '''
        return self.cats2levels

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return len(self.data[0])

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return self.data.shape[0]

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers` list.
        '''
        indices_list = []

        for i in range(len(headers)):
            if headers[i] in self.header2col:
                indices_list.append(list(self.header2col).index(headers[i]))

        return indices_list


    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself. This can be accomplished with numpy's copy
            function.
        '''
        return self.data.copy()

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        return self.data[0:5,:]

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        if self.data.shape[0] > 5:
            return self.data[self.data.shape[0]-5:self.data.shape[0],:]
        else:
            return self.get_all_data()

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        self.data = self.data[start_row:end_row,:]
        

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return column #2 of self.data.
        If rows is not [] (say =[0, 2, 5]), then we do the same thing, but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select. Empty list [] means take all rows.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''
        #using the self.get_header_indices to make the code more efficient
        index_list = self.get_header_indices(headers)
        if len(rows) == 0:
            new_arr = self.data[:,index_list]
        else:
            new_arr = self.data[np.ix_(rows,index_list)]
        return new_arr