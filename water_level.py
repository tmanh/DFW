import logging
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from common.utils import instantiate_from_config
from model.attention import MLPAttention
from model.mlp import *

from dataloader import *
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def plot_loss_chart(losses, title="Training Loss Over Epochs", save_path='loss.png'):
    """
    Plot a chart for loss values over epochs.

    Parameters:
        losses (list or array-like): List of loss values.
        title (str): Title of the chart.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linestyle='-', label='Training Loss')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=600)
        # print(f"Graph saved to {save_path}")

    # Show the plot
    # plt.show()
    plt.close()


def plot_predictions_with_time(predictions, save_path=None):
    """
    Plots the predicted values against their corresponding timestamps.

    Args:
        predictions (list): List of predicted values.
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.
    """ 
    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(predictions)), predictions, linestyle='-', label='Predicted Values')
    plt.title('Predicted Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('Predicted Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=600)
        # print(f"Graph saved to {save_path}")

    # Show the plot
    # plt.show()
    plt.close()


def location_aware_loss(o, y, loc_list):
    """
    Calculate the loss based on query points, grouping by GPS locations.

    Args:
        o (torch.Tensor): Predicted output, shape (B, ...)
        y (torch.Tensor): Ground truth, shape (B, ...)
        loc_list (torch.Tensor): GPS location of the query points, shape (B, 2) or (B, N, 2)

    Returns:
        torch.Tensor: Location-aware loss.
    """    
    # Compute per-sample loss
    loss_list = ((o - y) ** 2).mean(dim=1)  # Mean squared error per sample, shape (B,)

    # Group losses by unique GPS locations
    unique_locs, loc_indices = torch.unique(loc_list, dim=0, return_inverse=True)
    loss_per_location = torch.zeros(len(unique_locs), device=o.device)
    counts = torch.zeros(len(unique_locs), device=o.device)

    # Sum losses and counts for each unique location
    for i, loc_index in enumerate(loc_indices):
        loss_per_location[loc_index] += loss_list[i]
        counts[loc_index] += 1

    # Normalize losses by the number of occurrences for each location
    mean_loss_per_location = loss_per_location / counts

    # Compute the overall loss as the mean across locations
    overall_loss = mean_loss_per_location.mean()

    return overall_loss


def create_model(model_size, inputs):
    if inputs == 'p':
        in_dim = 20
    elif inputs == 'pt':
        in_dim = 21
    elif inputs == 'ptn':
        in_dim = 22
    elif inputs == 'pte':
        in_dim = 28
    elif inputs == 'ptev':
        in_dim = 29
    elif inputs == 'shortv':
        in_dim = 8
    elif inputs == 'short':
        in_dim = 7
    elif inputs == 'update':
        in_dim = 11


def test(n_points, cfg, train=True, testing_train=False):
    noise = True if 'n' in cfg.dataset.inputs else False

    with open('data/processed.pkl', 'rb') as f:
        data = pickle.load(f)
        all_nb = data.keys()

    # median
    good_nb = [(50.9290002709843, 4.05236274962837), (50.9298444159499, 4.05204412856079), (50.9306173352132, 4.05198202364574), (50.9297725404948, 4.05205883722317), (50.9300867420442, 4.05190024958134), (50.9294760876743, 4.05213195620184), (50.9456111983981, 4.03874724427939), (50.92956813631646, 4.230093815916902), (50.93254123135264, 4.22809352656834), (50.94327258107145, 4.226539047654562), (50.94670856428402, 4.228278925934652), (50.7982056056857, 3.750674374665007), (50.9200786511952, 4.06002871846776), (51.0127354493679, 4.58419569337913), (50.99086061703545, 4.581616507878948), (50.99797064815616, 4.510579523485871), (50.79377703793274, 4.269836627267547), (50.78349430046708, 4.243043165619314), (50.78387287610129, 4.243992247136886), (50.81396345407323, 3.9135086692261), (50.71400717195589, 4.229555139210667), (50.80455185633217, 4.288597225055614), (50.80412948406733, 3.871624994308589), (50.954166684563, 4.19564030626665), (50.998289571452, 4.30232652526809), (51.0129303709821, 4.34406660436635), (50.7996933846169, 4.07143226948205), (50.9796405088383, 4.70088527282975), (50.9305561605679, 4.57036934998288), (50.9353250059234, 4.58381035219212), (50.8217047246566, 4.19604643977163), (50.8240034459497, 4.20715091604736), (50.7334668664643, 4.22962514475058), (50.7346614972846, 4.32122013582505), (51.01908525043023, 4.101066416104875), (51.04583897998156, 3.384946611304884), (51.03393855766628, 3.551301116668707), (51.09635185294309, 4.319206870438431), (51.08425634279519, 4.350680939008726), (50.91073852974638, 3.419547724481932), (50.89032845015782, 3.433353503583627), (50.88238239426254, 3.43678590339766), (50.87563244603118, 3.428645694752737), (50.92126661788394, 3.413753809080222), (50.94556010096859, 3.385894353374373), (50.93104509480826, 3.348056456369687), (50.88498298943323, 3.693711150797316), (50.88303533138356, 3.716709589855025), (50.87907571883665, 3.742391593839773), (50.86176331477927, 3.765947100079222), (50.87329717351446, 3.692777772476414), (50.84865217525669, 3.71206251112564), (50.8634364057813, 3.771243959792352), (50.83376576214968, 3.76898864658583), (50.80790167884659, 3.626698144970669), (50.75330774120651, 3.963098792810351), (50.75034473379703, 3.979828442354752), (50.7831912869604, 4.20824574998637), (50.7437574353902, 4.007515530656623), (50.912244189585, 4.60880646868609), (50.7486391043145, 4.35302730688102), (50.9576858938692, 4.21119283279234), (50.73395786448076, 4.023726457277386), (50.72412835294058, 4.025880166345091), (50.7345603967519, 4.320725008842), (50.69875568962166, 4.012782691358225), (50.95597487475767, 4.025802537279502), (50.95274524107715, 4.016739231957115), (50.9622479726803, 4.71994448572574), (50.9594407958578, 4.72376650194768), (50.94884025884854, 3.999990812755693), (50.94707029210976, 3.991769859983676), (50.94011725684002, 3.978058812144063), (50.93052730927617, 3.972397349886937), (50.8862853448484, 4.095096517154569), (50.84846416289579, 4.132176381269662), (50.89699336879984, 4.155412289191527), (50.87967260654746, 4.170914954977651), (50.88525912915445, 4.187410416783746), (50.86953283381866, 4.215836544543968), (50.9769218831948, 4.26340138257842), (50.9736582973071, 4.26282491873049), (50.86918049452461, 4.21244667582111), (50.9769961603781, 4.2660923530296), (50.88760559711391, 3.946918835588328), (50.88294639417263, 3.924252177267686), (51.02323242118194, 4.104480583178382), (50.88070814784588, 4.153957051838565), (50.7915540177442, 3.62257299330959), (50.80952074433264, 4.29753869861233), (50.76482189211035, 4.271663187398281), (51.00843692787139, 4.527206463890328), (50.94653737889095, 4.702372468590229), (51.03335045519115, 4.510054255592118), (50.86985198339722, 4.694082872499124), (50.81741663796304, 4.634646775389621), (50.88623399834572, 4.699073946658467), (50.8719935905382, 4.691048886797695), (50.88189029592182, 4.697146286757016), (50.85188373352406, 4.661719075787183), (51.002247, 4.081777), (50.9390455726714, 4.74065635521849), (50.85857753755359, 4.670146375566242), (50.967587, 3.46339), (50.812199, 3.61216), (50.751148, 4.257219999999999), (50.938168, 4.734415), (50.917759000000004, 3.95716), (50.902378000000006, 3.962532), (50.94262907125425, 3.372612595562544), (50.752506, 3.6088760000000004), (50.8190208479953, 3.67114141932069), (50.8260145913424, 3.61160845853722), (50.8124284499878, 3.61266412679552), (50.7933984258841, 3.74097901823192), (50.9067525913702, 3.57874295119978), (50.9111191595063, 3.58643343804359), (50.808243, 4.276992), (50.827263, 3.603416), (50.79676743, 3.82899239), (50.95700321, 3.588745537), (50.76505468, 3.820099105), (50.99202123, 3.97381195), (50.81312026, 3.679699711), (50.86819925, 4.012681623), (50.92265198, 3.461885894), (51.03879005, 4.006852561), (51.03667208, 3.524172457), (50.84022646, 3.99905047), (51.05206974, 3.504329223), (50.89711467, 3.601662218), (50.82994653, 4.056235337), (50.75281561, 3.632510695), (50.86571265, 3.993580156), (50.81878745, 3.847756318), (50.893148, 3.480404), (50.880082, 3.488962), (50.878875, 3.48992), (50.909992, 3.477923), (50.803914, 3.961181), (50.987125, 3.39075), (50.992447, 3.2775730000000003), (51.07585573646379, 3.621395341040125), (50.93096933327577, 4.351029326696788), (50.883953000000005, 4.130686), (50.880013, 3.734605), (50.807453, 3.95696), (51.04855660748743, 4.448811229198293), (50.87379375663088, 4.697672087116771), (50.792393, 3.859385), (51.0217277576886, 4.26818982755525), (50.816715, 4.02474), (50.9389947336077, 4.1147821211541), (50.9313481365844, 4.02412437200653), (50.922974441276, 3.9660580046409), (50.851636834058, 3.56048211971862), (50.8656215307883, 3.98770030112802), (50.9379009571661, 3.44986917244354), (50.9990233913782, 3.95541926476604), (50.879909, 3.781667), (50.8119436632244, 3.77711323015109), (50.97837371264585, 3.8790847082488), (51.0193776263799, 4.53492959549705), (51.029174, 3.5204014), (51.006583, 3.652972), (51.024963, 4.641819), (51.079815, 4.465216), (51.0950639708226, 4.31758089338274), (50.9827682912497, 4.18529296790663), (50.924686, 4.71477), (50.9855693760846, 4.4913586067665), (50.7877505084434, 4.58314835748828), (50.7972443822579, 4.64296518846761), (50.88332, 3.708224), (50.877827, 3.690515), (50.9604026077283, 4.65738640613315), (50.857426, 4.761357), (50.9363141119176, 4.37439807905561), (50.987865, 4.427994), (50.936028, 4.354853), (51.00145, 4.445373), (50.926899, 4.405815), (50.9812741981779, 4.74634213201572), (50.987656, 4.304885), (50.980370902042, 4.26972108793651), (51.005112, 4.299315), (51.031685, 4.283012), (50.975487, 4.309776), (50.864609, 4.677224), (50.862137, 4.659367), (50.882305, 4.697023), (51.0173847, 4.681001), (50.926463, 4.588865), (50.875225, 4.694857), (50.745453, 4.345417), (50.730686, 4.30414), (50.776932, 4.287518), (50.77988, 4.325822), (50.984913, 4.156368), (50.991325, 4.360341), (50.88311086258833, 4.699775251046551), (50.85760244903668, 4.670387235742362), (51.04492931928172, 4.280674405365077), (51.03256435, 4.26339105), (50.95456921671166, 4.234859153338897), (50.95046776583727, 3.272193441564975), (50.88541834624587, 3.686130099317326), (50.94754787056984, 3.638493838397924), (50.87136999674804, 3.659283091129403), (50.88241865994927, 3.556584873524041), (50.98276829123967, 4.185292967906626), (50.92331889865253, 3.96689419479605), (50.87282736801505, 4.056547787684996), (50.88874722238768, 4.116261134168793), (50.87655655548213, 4.156792081494812), (50.87603247711527, 4.176160287733671), (50.77234960526412, 3.871449700827389), (50.74133825915049, 3.891852475717929), (50.81112752773524, 3.901021486015226), (50.9425818026234, 4.062032087594483), (50.9252606469659, 4.7063826186718), (50.83796678059495, 4.643260073189932), (50.80289998164054, 4.642501713574058), (50.98643160991459, 4.569932353954903), (50.97723224447928, 4.495280851316309), (50.87147680652693, 4.690353455347827), (50.85840785315725, 4.615556171727959), (50.90573931120411, 4.711303315338729), (50.86707494423172, 4.6989602849318), (50.86028673679752, 4.738426897418766), (50.76857826444854, 4.604976634469243), (50.79015924494613, 4.592987725977424), (50.8228698761891, 4.640844210446453), (50.76569220179697, 4.279775129451005), (50.9456273918501, 4.40122490688286), (50.77844062107208, 4.21186328740738), (50.9400341540623, 4.737889793297333), (50.94335838649002, 4.744133973852081), (50.9761639541252, 3.50350061937995), (50.7620116683951, 3.86733483005445), (50.9376347849076, 4.04643624264415), (51.1069984381704, 3.43509535017227), (50.9148702570547, 3.6705370384779), (50.9878959801345, 3.52335264918976), (50.981891874827, 3.53357211241346), (51.0001626403935, 4.07806732915739), (50.8736649491979, 4.07715178150251), (50.9296307282077, 3.6542880951664), (50.7695564802607, 3.88066612268998), (50.7377444522776, 4.2420029010962), (50.8502180339115, 3.3001197904026), (50.7080525616338, 4.2202229496191), (50.7658454688447, 4.26908691804533), (51.0899074227115, 3.56287609346259), (50.8296965118308, 4.00959912439749), (50.7844688516595, 4.28839005991179), (50.9134704825591, 3.4288159529755), (50.9093479751093, 3.40378795033366), (51.0223304151673, 3.6396663123729), (50.9176726362496, 4.4205596749182), (51.1011257965587, 3.56546697228752), (50.751208179845, 3.971109149875746), (50.83932141964262, 3.929491089697852), (50.91989179799585, 4.242722510241078), (50.91791115773739, 4.23995472415737), (50.86674841185019, 4.166666352167602), (50.8598332535142, 4.16023345956444), (50.85660437415456, 4.204644353273731), (50.86199943760177, 4.212253134868256), (50.99012037214969, 4.619584598557898), (50.9676756237952, 4.61472358499743), (51.008611349865, 4.72693259436862), (50.9771622425665, 4.66408298012881), (50.9719359409419, 4.562281445861772), (50.9974231487054, 4.55493692805425), (50.97966511885235, 4.513542669042173), (50.93121918763402, 4.341157147208036), (50.7535570834951, 4.26976461821304), (50.74127804477154, 4.336108626639918)]
    test_nb = [(50.7437574353902, 4.007515530656623), (50.8736649491979, 4.07715178150251), (50.9067525913702, 3.57874295119978), (50.83796678059495, 4.643260073189932), (50.76857826444854, 4.604976634469243), (50.98276829123967, 4.185292967906626), (50.86176331477927, 3.765947100079222), (50.88311086258833, 4.699775251046551), (50.86571265, 3.993580156), (50.792393, 3.859385), (50.8190208479953, 3.67114141932069), (50.730686, 4.30414), (50.88070814784588, 4.153957051838565), (50.87603247711527, 4.176160287733671), (50.9300867420442, 4.05190024958134), (50.91791115773739, 4.23995472415737), (50.7915540177442, 3.62257299330959), (50.86199943760177, 4.212253134868256), (50.79015924494613, 4.592987725977424), (51.005112, 4.299315), (50.9878959801345, 3.52335264918976), (50.7486391043145, 4.35302730688102), (50.75034473379703, 3.979828442354752), (50.91073852974638, 3.419547724481932), (51.0950639708226, 4.31758089338274), (50.87329717351446, 3.692777772476414), (50.87379375663088, 4.697672087116771), (50.9390455726714, 4.74065635521849), (51.03879005, 4.006852561), (50.9200786511952, 4.06002871846776), (50.880013, 3.734605), (50.87907571883665, 3.742391593839773), (50.912244189585, 4.60880646868609), (50.987125, 3.39075), (50.99797064815616, 4.510579523485871), (50.75330774120651, 3.963098792810351), (50.89032845015782, 3.433353503583627), (51.0217277576886, 4.26818982755525), (50.902378000000006, 3.962532), (50.94754787056984, 3.638493838397924), (50.8119436632244, 3.77711323015109), (50.8260145913424, 3.61160845853722), (50.90573931120411, 4.711303315338729), (50.77988, 4.325822), (50.94707029210976, 3.991769859983676), (50.776932, 4.287518), (50.94670856428402, 4.228278925934652), (50.922974441276, 3.9660580046409), (50.87967260654746, 4.170914954977651), (50.8656215307883, 3.98770030112802), (51.03667208, 3.524172457), (51.04855660748743, 4.448811229198293), (50.92126661788394, 3.413753809080222), (50.99086061703545, 4.581616507878948)]
    train_nb = [f for f in all_nb if f not in test_nb]
    train_good_nb = [f for f in good_nb if f not in test_nb]

    # 🔹 1️⃣ Define Device (Multi-GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = WaterDataset(
        path='data/processed.pkl', train=True,
        selected_stations=train_good_nb, input_type=cfg.dataset.inputs
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # 🔹 2️⃣ Initialize Model
    model = instantiate_from_config(cfg.model).to(device)

    # 🔹 3️⃣ Enable Multi-GPU Support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # Use torch_geometric.nn.DataParallel if necessary
    model = model.to(device)

    l1_loss = nn.L1Loss()
    l1_smooth = nn.SmoothL1Loss()

    print('Config:', cfg.model.params.fmts)

    # Training loop
    not_finish_training = True
    stage = 1
    num_epochs = 25
    freeze_base = False

    n_loops = 1
    if 'w' in cfg.model.params.fmts:
        n_loops += 1
    if isinstance(model, MLPAttention):
        n_loops += 1

    while train and not_finish_training:
        if freeze_base:
            if torch.cuda.device_count() > 1:
                model.module.freeze()
            else:
                model.freeze()

        list_loss = []
        epoch_bar = tqdm(range(num_epochs), desc="Epochs")  # Initialize tqdm for epochs
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in epoch_bar:
            for x, xs, y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
                x = x.to(device)
                xs = xs.to(device)
                y = y.to(device)

                B, S, M, N = x.shape
                x = x.view(B * S * M, N)
                xs = xs.view(B * S * M, N, -1)
                y = y.view(-1, 1)

                # Forward pass
                o = model(xs, x, inputs=cfg.dataset.inputs, train=True, stage=stage)

                # Compute L1 loss on all elements
                loss = l1_loss(o, y.repeat((1, o.shape[1])))

                if 'i' in cfg.dataset.inputs:
                    repeated_o = o.repeat((1, x.shape[1]))
                    ixs = get_i_inputs(xs.detach(), repeated_o.detach(), cfg.dataset.inputs)
                    io = model(ixs, repeated_o, inputs=cfg.dataset.inputs, train=True)

                    # Compute loss for i-inputs (optional)
                    loss = loss + l1_loss(io, x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print loss for every epoch
                epoch_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
                list_loss.append(loss.item())

            # Save model checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"model_{epoch}.pth")

            # Plot loss trend
            plot_loss_chart(list_loss)

        n_loops -= 1

        if n_loops <= 0:
            not_finish_training = False
        
        if isinstance(model, MLPAttention) and not freeze_base:
            model.using_attention = True
            freeze_base = True
        elif 'w' in cfg.model.params.fmts:
            stage = 2
            freeze_base = True

    # TODO
    # model.load_state_dict(torch.load("model_24.pth"), strict=False)
    model.eval()

    total_error = 0
    errors = []
    testing_set = train_good_nb if testing_train else test_nb
    for tnb in testing_set:  # train_good_nb, test_nb:
        batch_size = 10
        test_dataset = WaterDataset(
            path='data/processed.pkl', train=False,
            selected_stations=[tnb], input_type=cfg.dataset.inputs
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        n_sample = 0
        list_l1 = []
        with torch.no_grad():
            for idx, (x, xs, y) in enumerate(test_loader):
                if idx >= 20:
                    break
                B, N, C = xs.shape
                x = x.to(device)
                xs = xs.to(device)
                y = y.to(device)

                # sv, dt, w1 / 66, w2 / 66, distance, bidistance, *dp, h1, h2, mean_w, std_w, max_zw
                # o = torch.sum(x * torch.softmax(torch.softmax(1 / (xs[:, :, 1] + 1e-7), dim=1) * xs[:, :, 4], dim=1), dim=1, keepdim=True)
                o = model(xs, x, inputs=cfg.dataset.inputs, train=False, stage=-1)
    
                l1_elements = torch.abs(o - y)
                l1_elements = torch.mean(l1_elements, dim=1)

                loss = torch.sum(l1_elements)

                list_l1.extend(list(l1_elements.detach().cpu().numpy()))

                xs = xs.detach()
                o = o.unsqueeze(1).repeat(1, xs.shape[1], 1)

                total_loss += loss.item()
                n_sample += x.shape[0]

        l1 = total_loss / n_sample
        total_error += l1
        errors.append(l1)
        print(total_loss / n_sample, tnb, np.mean(list_l1), np.std(list_l1))

    errors = np.array(errors)

    print('Mean error: ', total_error / len(testing_set))
    percentiles = [30, 40, 50, 60, 70, 80, 90, 100]

    for p in percentiles:
        # Compute the threshold value at the p-th percentile
        threshold = np.percentile(errors, p)
        
        # Get the subset of errors that are below this threshold
        subset = errors[errors <= threshold]
        
        # Since thresholding by percentile guarantees that ~p% of values are included,
        # fraction should be approximately p/100
        fraction = len(subset) / len(errors)
        mean_val = np.mean(subset)
        
        logging.info(f"Lowest {p}% (threshold: {threshold:.4f}): fraction = {fraction:.3f}, mean = {mean_val:.4f}")

    train_or_test = 'Training data' if testing_train else 'Testing data'
    print(f'Number of points: {n_points} - {train_or_test} - Mean error: {total_error / len(testing_set)}\n')

    return l1


def main():
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    random.seed(42)  # For reproducibility

    parser = argparse.ArgumentParser(description="Run the test function with configurable parameters.")
    parser.add_argument("--cfg", type=str, help="Config file path", default="config/mlp_32_1_wo.yaml")
    parser.add_argument("--training", action="store_true", help="Enable training mode (default: False)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    if args.training:
        print('-----Training-----')
        test(16, cfg, train=True)
        
    print('-----Testing-----')
    # test(2, cfg, train=False, testing_train=True)
    # test(4, cfg, train=False, testing_train=True)
    # test(6, cfg, train=False, testing_train=True)
    # test(8, cfg, train=False, testing_train=True)
    # test(10, cfg, train=False, testing_train=True)
    
    print('-----Testing-----')
    # test(2, cfg, train=False)
    # test(4, cfg, train=False)
    # test(6, cfg, train=False)
    # test(8, cfg, train=False)
    test(10, cfg, train=False)


def save_results(out, l1, list_l1, list_values):
    logging.info(f'{out} {l1.item()}\n')
    print(f'{out} {l1.item()}')

    # Example usage
    # Assuming `times` and `list_values` are already populated from your code
    plot_predictions_with_time(list_l1, save_path=f'{out}-l1.png')
    plot_predictions_with_time(list_values, save_path=f'{out}-series.png')


main()
