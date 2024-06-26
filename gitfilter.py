import os
import pathspec

# Class to define what files the program should process
# includes the .gitignore file in a project as well as 
# a pass filter on custom filetypes

# file pass filter by extension.
FILE_INCLUDE_FILTER = ['.cs', '.cshtml', '.py']

# service to find the gitignore file, read its rules and add custom rules alongside
# will also be used to test if a certain file / directory should be processed
class GitFilterService:
    def __init__(self, root_path):
        self.root_path = root_path
        self.git_directory = self.find_git_directory()
        self.git_ignore_rules = self.read_git_ignore()
        self.file_include_rules = self.add_custom_rules()

    # look for a gitignore file and set our gitignore directory
    def find_git_directory(self):
        # search for gitignore
        for root, dirs, files in os.walk(self.root_path):
            if ".gitignore" in files:
                return os.path.join(root, ".gitignore")

    # read the gitignore files and store the rules
    def read_git_ignore(self):
        if self.git_directory is None:
            return None

        git_ignore_rules = list()
        with open(self.git_directory, 'r') as f:
            git_lines = f.readlines()
        for rule in git_lines:
            # skip comments in the gitignore
            if "#" in rule:
                continue
            new_rule = rule.replace('\n', '')
            if new_rule != "":
                git_ignore_rules.append(new_rule)
        return git_ignore_rules

    # add any custom rules (inclusive rule) for filetype
    def add_custom_rules(self):
        cust_rules = list()
        for rule in FILE_INCLUDE_FILTER:
            cust_rules.append(rule)
        return cust_rules

    # test a file / folder relative filepath to see if it is ignored by git or custom filter
    def git_filter_allow(self, rel_filepath):
        # windows convert path string from \\ to /
        file_path_linux = str(rel_filepath).replace("\\","/")
        # apply the git ignore lines to the file path
        spec = pathspec.PathSpec.from_lines("gitwildmatch", self.git_ignore_rules)
        if spec.match_file(file_path_linux):
            return False
        # check to see if the file has a valid file extension
        if "." in file_path_linux:
            file_extension = "." + file_path_linux.rsplit(".", maxsplit=1)[-1]
            if file_extension not in self.file_include_rules:
                return False
        return True
