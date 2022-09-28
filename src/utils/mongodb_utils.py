import io
import getpass
import urllib.parse
import pathlib
import pymongo
import gridfs


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def get_auth_info():
    import os

    this_file_dir = os.path.dirname(os.path.abspath(__file__))

    db_file = pathlib.Path(os.path.join(this_file_dir, ".db"))
    if db_file.is_file():
        print('Reading authentication info from ".db" file')
        with db_file.open("r") as f:
            user, password, address, db_name = f.read().splitlines()

    else:
        print(
            "Please type the authentication information below.\n"
            'You may also set ".db" file with:\n'
            "{username}\n{password}\n{address:port}\n{db name}"
        )
        user = input("Username: ")
        password = getpass.getpass()
        address = input("address:port:")
        db_name = input("db_name")

    user = urllib.parse.quote_plus(user)
    password = urllib.parse.quote_plus(password)

    return user, password, address, db_name


def get_db_uri():
    user, password, address, db_name = get_auth_info()
    ip, port = address.split(":")

    return f"mongodb://{user}:{password}@{ip}/{db_name}"


def get_db():
    user, password, address, db_name = get_auth_info()
    ip, port = address.split(":")

    client = pymongo.MongoClient(
        f"mongodb://{user}:{password}@{ip}/{db_name}", int(port)
    )
    db = client[db_name]

    return db


def get_experiment_artifact(artifact_name, experiment_query):
    db = get_db()
    experiment = db.runs.find_one(experiment_query, sort=[("_id", pymongo.DESCENDING)])
    if experiment is None:
        print(f"Could not find experiment with query {experiment_query}")
        return None

    print(f'Selected artifact from experiment {experiment["_id"]}')
    artifact = next(
        (x for x in experiment["artifacts"] if x["name"] == artifact_name), None
    )
    if artifact is None:
        print(f"Could not find artifact with name {artifact_name}")
        return None

    object_id = artifact["file_id"]
    fs = gridfs.GridFS(db)
    model = fs.get(object_id).read()

    model = io.BytesIO(model)
    return model
