import os


def get_out_dir():
    file_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(
        file_path)))
    out_path = os.path.join(project_path, 'out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    return out_path
