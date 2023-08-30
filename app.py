from flask import Flask, render_template, url_for, request, send_file, abort
import subprocess as sp
from time import sleep
import requests as requests
from flask_sock import Sock

from files_management import *

app = Flask(__name__)  # app Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1204  # 8 MB max file size / error 413
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}  # 'max_message_size': 10240}  # websocket options
app.config['MAX_NUM_FILES'] = 10  # Max number of files that can be elaborated
app.config['RECAPTCHA_SITE_KEY'] = ""  # Site key for g-recaptcha v3
app.config['RECAPTCHA_SECRET_KEY'] = ""  # Secret key for g-recaptcha v3
app.config['RECAPTCHA_VERIFY_URL'] = "https://www.google.com/recaptcha/api/siteverify"  # g-recaptcha verify url
sock = Sock(app)  # websocket initialization


@app.route('/', methods=['GET'])
def index():
    """
    Path: / .
    Index page. Here there is the form to submit a task.
    Methods allowed: GET

    Returns
    -------
    str
        Index page
    """
    return render_template('index.html', site_key=app.config['RECAPTCHA_SITE_KEY'])


@sock.route('/sock')
def sock(ws):
    """
    Handles websocket connections and requests.
    Here is a sequence diagram:

    Parameters
    ----------
    ws : simple_websocket.Server
        Object websocket server

    Returns
    -------
    str
        page with no meaning, should be ignored
    """
    #  See Also
    #     --------
    #     Sock: flask_sock library here used
    unique = ""
    try:
        unique = ws.receive(timeout=5)
    except Exception:
        ws.close(reason=1002, message="TIME OUT EXPIRED")

    if check_unique(unique):
        if check_request(unique):
            if remove_from_unique_not_used(unique):
                try:
                    # execution of NMTF-link script
                    # Create a subprocess (args, ...), stdout=sp.PIPE -> create a sp.PIPE object for subprocess stdout,
                    #  text=True -> set output to be a stream of new lines and not bytes
                    nmtf = sp.Popen(['python', '-u', 'NMTF_link.py', get_path(unique), get_standard_setting_file_name()],
                                    stdout=sp.PIPE,
                                    text=True,
                                    bufsize=-1)
                    out = nmtf.stdout
                    print(unique + " Before nmtf webs")
                    ws.send("nmtf app lunched")
                    print(unique + " nmtf app lunched")
                    while nmtf.poll() is None:
                        line = out.readline().strip()
                        if line != "":
                            ws.send(line)
                            sleep(0.2)
                        else:
                            sleep(2)
                    print(unique + " polling nmtf finished")
                    ws.send("polling nmtf finished")
                    nmtf.wait()
                    print(unique + " nmtf finished")
                    for line in out.readlines():
                        line = line.strip()
                        if line != "":
                            ws.send(line)
                    if nmtf.returncode == 0:
                        print(unique + " All finished nmtf well")
                        ws.close(1000, message="FILES_READY")
                    else:
                        ws.send("⚠️App return code not 0, please check your files!")
                        print(unique + " Subprocess return code not 0, unique: " + unique)
                        ws.close(reason=1011, message=f"ANALYSIS FAILED")
                except Exception:
                    ws.send("⚠️App return code not 0, please check your files!")
                    print(unique + " Exception occurred during execution of subprocess, unique: " + unique)
                    ws.close(reason=1011, message=f"ANALYSIS FAILED")

                add_to_heap(unique)  # After results are created I add filename to heap

            else:
                ws.close(reason=1016, message=f"TOO MUCH TIME")
        else:
            ws.close(reason=1008, message=f"ELABORATION WAS PERFORMED")
    else:
        ws.close(reason=1008, message=f"BAD REQUEST")


#  Helps to create setting file when configured from form
def create_set_file(req, files):
    """
    Create the setting file from a submitted form

    Parameters
    ----------
    req : ImmutableMultiDict[str, str]
        form submitted
    files : list
        names of association files

    Returns
    -------
    unique : str
        unique created to save the files.
    error : None
        something went wrong in file creation. Unique was created, but is added to not used

    Raises
    ------
    KeyError
        a parameter from the form is missing. The form is not consistent

    IndexError
        a parameter from the form is missing. The form is not consistent
    """
    fileString = "---\n"
    strategy = req["integration.strategy"]
    fileString += "  integration.strategy: " + strategy + "\n"
    initialization = req["initialization"]
    fileString += "  initialization: " + initialization + "\n"
    metric = req["metric"]
    fileString += "  metric: " + metric + "\n"
    iterations = req["number.of.iterations"]
    fileString += "  number.of.iterations: " + iterations + "\n"
    masking = req["type.of.masking"]
    fileString += "  type.of.masking: " + masking + "\n"
    stop = req["stop.criterion"]
    fileString += "  stop.criterion: " + stop + "\n"
    threshold = req["score.threshold"]
    fileString += "  score.threshold: " + threshold + "\n"
    k_svd = req["k_svd"]
    fileString += "  k_svd: " + k_svd + "\n"

    dsnames = req.getlist("dsnames")
    ks = req.getlist("ks")
    fileString += "  ranks:\n"
    for i in range(len(dsnames)):
        fileString += "    - dsname: " + dsnames[i] + "\n"
        fileString += "      k: " + ks[i] + "\n"

    leftNodes = req.getlist("nodes.left")
    rightNodes = req.getlist("nodes.right")
    main = req["main"]
    print("main", main)
    fileString += "  graph.datasets:\n"
    for i in range(len(files)):
        fileString += "    - nodes.left: " + leftNodes[i] + "\n"
        fileString += "      nodes.right: " + rightNodes[i] + "\n"
        fileString += "      filename: " + files[i] + "\n"
        fileString += "      main: "
        if int(main) == i:
            fileString += "1"
        else:
            fileString += "0"
        fileString += "\n"
    print(fileString)
    unique = new_unique_name()
    create_dir(unique)
    try:
        f = open(get_path(unique) + get_standard_setting_file_name(), "w")
        f.write(fileString)
        f.close()
    except OSError:
        print("Problems with setting file creation")
        add_to_unique_not_used(unique)
        return None

    return unique


@app.route('/loader', methods=['POST'])
def loader():
    """
    Path: /loader .
    Page to witch the form is submitted. Handles form checking and captcha checking.
    Methods allowed: POST

    Returns
    -------
    page : str
        loader.html page. This page has the prompt where the websocket communication prints the status of the task.
    page_error : str
        If the form is not consistent an error page 400 is returned. If test recaptcha failed error page 401 is
        returned.
    """
    # Uncomment this part and configure the app.config['RECAPTCHA_SECRET_KEY'] and app.config['RECAPTCHA_SITE_KEY']
    # to activate the recaptcha test
    # recaptcha_test = requests.post(url=f"{app.config['RECAPTCHA_VERIFY_URL']}" +
    #                                   f"?secret={app.config['RECAPTCHA_SECRET_KEY']}" +
    #                                   f"&response={request.form['g-recaptcha-response']}").json()
    if True:  # recaptcha_test["success"]:
        unique = ""
        set_file = request.files["sfile"]
        files = request.files.getlist("afiles")
        try:
            if len(files) <= app.config['MAX_NUM_FILES']:
                if request.form["how_sfile"] == "1":
                    if check_config_file(set_file):
                        if check_files(files):
                            unique = new_unique_name()
                            create_dir(unique)
                            save_file(set_file, unique, is_setting=True)
                            save_files(files, unique)
                            add_to_unique_not_used(unique)
                        else:
                            # Error in association files
                            print("Error in association files")
                            abort(400)
                    else:
                        # Error in setting file
                        print("Error in setting file")
                        abort(400)
                else:
                    filesNames = list(map(lambda f: f.filename, files))
                    unique = create_set_file(request.form, filesNames)
                    if unique is not None and check_files(files):
                        save_files(files, unique)
                        add_to_unique_not_used(unique)
                    else:
                        # Error in creating the setting file
                        print("Error in creating the setting file")
                        abort(400)
            else:
                # Error max files number exceeded
                print("Error max files number exceeded")
                abort(400)
        except KeyError:
            print("Key Error occurred")
            abort(400)
        except IndexError:
            print("Index Error occurred")
            abort(400)
    else:
        abort(401)
    return render_template('loader.html', unique=unique)


#
#  Name page: showdata
#  Methods: GET
#  Return: webpage
#  Template: data.html
#  Description: Plotting of the results in different types of graphs
#
@app.route('/showdata', methods=['GET'])
def showdata():
    """
    Path: /showdata .
    Page showing results of the task. Unique is passed as an argument in the link (name)
    Methods allowed: GET

    Returns
    -------
    page : str
        data.html page
    page_error : str
        Error page 400 if unique doesn't exist
    """
    u_name = request.args["name"]
    if check_unique(u_name):
        return render_template('data.html', unique=u_name, imagelist=get_images_list(u_name))
    else:
        abort(400)


#
#  Name page: showtxt
#  Methods: GET
#  Return: file
#  Description: Download of user's results
#
@app.route('/showtxt', methods=['GET'])
def showtxt():
    """
    Path: /showtxt .
    Api to get file txt with results.
    Arguments: 'name' for the unique and 'as_attachment'. Set as_attachment to 'yes' to download the file.
    Methods allowed: GET

    Returns
    -------
    page : str
        file txt as page or file as attachment
    page_error : str
        Error page 400 if unique doesn't exist
    """
    u_name = request.args["name"]
    as_attachment = request.args["as_attachment"] == "yes"
    if check_unique(u_name):
        fres = get_path_results(u_name)
        return send_file(fres, as_attachment=as_attachment,
                         download_name='your_results.txt')
    else:
        abort(400)


@app.route('/showimg', methods=['GET'])
def showimg():
    """
    Path: /showimg .
    Api to get a result image.
    Arguments: 'name' for the unique, 'img' for the image name, and 'as_attachment'.
    Set as_attachment to 'yes' to download the file.
    Methods allowed: GET

    Returns
    -------
    page : str
        image as page or image as attachment
    page_error : str
        Error page 400 if unique or image doesn't exist
    """
    u_name = request.args["name"]
    img_name = request.args["img"]
    as_attachment = request.args["as_attachment"] == "yes"
    if check_unique(u_name):
        try:
            imgres = get_path_img_res(u_name, img_name)
            return send_file(imgres, as_attachment=as_attachment)
        except AttributeError:
            abort(400)
    else:
        abort(400)


if __name__ == "__main__":
    # Code to execute once:
    routine_clean_heap()  # Starts clean heap routine
    routine_clean_uniques_not_used()  # Starts clean uniques not used routine
    app.run(host="0.0.0.0")
    index()
