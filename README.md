---

# MLP Implementation

This project is a university practice I completed in 2024 to implement a Multilayer Perceptron (MLP) for the MNIST dataset.

## Setup

Before using this code, you'll need to set up a virtual environment and install dependencies. Follow these steps:

1. **Open Your Terminal or Command Prompt:**
   Depending on your operating system, open Command Prompt (Windows) or Terminal (macOS, Linux).

2. **Navigate to the Directory Where You Want to Create the Virtual Environment:**
   Use the `cd` command to change directories. For example:
   ```
   cd ~
   ```

3. **Create the Virtual Environment:**
   Once you're in the desired directory, run the following command to create a virtual environment named "MLP":
   ```
   python -m venv MLP
   ```

   This command will create a new directory named "MLP" which will contain the virtual environment files.

4. **Activate the Virtual Environment:**
   After creating the virtual environment, activate it with the appropriate command based on your operating system:
   - **Windows:**
     ```
     MLP\Scripts\activate
     ```
   - **macOS and Linux:**
     ```
     source MLP/bin/activate
     ```

5. **Install Dependencies:**
   With the virtual environment activated, install the project dependencies listed in the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

   This command will install all the required packages listed in the `requirements.txt` file into your virtual environment.

## Running the Code

To run the code and see the results, follow these steps:

1. **Prepare Data:**
   The code requires the MNIST dataset. If you don't have the dataset already, don't worry! The code will automatically download and prepare the dataset for you.

2. **Run the Code:**
   Simply execute the `main.py` file using Python:
   ```
   python main.py
   ```

   This command will execute the main script, which loads the dataset using `data_loader.py`, trains the MLP model, and displays the results in the command line.

3. **View Results:**
   After running the code, you'll see the results in the command line interface. Additionally, the results will be saved as images in a directory named "result" in the same directory as the code.

## Workflow Explanation

- **Data Preparation:**
  The code uses `setup.py` to download and prepare the MNIST dataset. If the "data" directory doesn't exist, it will be created automatically.

- **Code Execution:**
  When you run `main.py`, the code loads the dataset using `data_loader.py`, trains the MLP model, and displays the results in the command line interface.

- **Result Visualization:**
  The results are also saved as images in a directory named "result" in the same directory as the code.

## Usage

Now that your virtual environment is activated, dependencies are installed, and the code is executed, you can analyze the results and further customize the code as needed.

To deactivate the virtual environment and return to your global Python environment, simply type:
```
deactivate
```

---

