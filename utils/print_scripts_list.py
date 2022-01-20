def print_scripts_list():
    with open('utils/scripts_list.txt') as file_id:
        scripts_list = file_id.read()
        print(scripts_list)