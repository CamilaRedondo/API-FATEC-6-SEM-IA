import os


def get_out_dir():
    file_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(
        file_path)))
    out_path = os.path.join(project_path, 'out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    return out_path


def get_project_dir():
    file_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(
        file_path)))
    return project_path


def get_logs_dir(log_type):
    file_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(
        file_path)))
    logs_path = os.path.join(project_path, 'logs')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    logs_full_path = os.path.join(logs_path, log_type)
    if not os.path.exists(logs_full_path):
        os.makedirs(logs_full_path)

    return logs_full_path
