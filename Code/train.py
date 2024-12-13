from collections import defaultdict
import datetime
import pathlib
import torch
import uniplot
from torch.utils.data import DataLoader
from model import (
    GapAheadAMTSRegressor,
    PositionalEncoding,
    index_agreement_torch,
    ModelConfig,
)

import sys
import yaml

sys.path.append("src")
from processing.loader import (
    SantosTestDatasetTorch,
)

from dataset import SantosTrainDatasetTorch


def main():
    NUM_LAYERS_RNN = 1
    NUM_LAYERS_GNN = 2
    TIME_ENCODING_SIZE = 32
    HIDDEN_UNITS = 64
    NUM_HEADS_GNN = 4

    model_config = ModelConfig(
        num_layers_rnn=NUM_LAYERS_RNN,
        num_layers_gnn=NUM_LAYERS_GNN,
        num_heads_gnn=NUM_HEADS_GNN,
        time_encoding_size=TIME_ENCODING_SIZE,
        hidden_units=HIDDEN_UNITS,
    )

    epochs = 50
    batch_size = 32
    context_increase_step = 60 * 12.0
    forecast_increase_step = 60 * 4.0
    sample_size_per_epoch = 1000
    train_data_path = "data/02_processed/train"
    test_data_path = "data/02_processed/test"
    lr = 0.0005
    model_version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    start_model_path = ""
    model_name = pathlib.Path(__file__).parent.name

    # sample_size_per_epoch = 50
    # start_model_path = "data/04_model_output/20240714032850/epoch_11.pt"

    out_path = f"data/04_trained_models/{model_name}/{model_version}"

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "model_config.yml", "w") as f:
        yaml.dump(model_config.to_dict(), f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not ((train_data_path := pathlib.Path(train_data_path)).exists()):
        raise FileNotFoundError(f"Train data path {train_data_path} does not exist")

    if not ((test_data_path := pathlib.Path(test_data_path)).exists()):
        raise FileNotFoundError(f"Test data path {test_data_path} does not exist")

    context_window_lengths = defaultdict(
        float,
        {
            "cattalini_corrente": 60 * 48.0,
            "cattalini_maregrafo": 60 * 48.0,
            "cattalini_meteorologia": 60 * 48.0,
            "odas_corrente": 60 * 48.0,
            "odas_meteorologia": 60 * 48.0,
            "porto_astronomica": 60 * 48.0,
            "porto_harmonico": 60 * 48.0,
            "porto_maregrafo": 60 * 48.0,
        },
    )

    max_context_window_lengths = defaultdict(
        float,
        {
            "cattalini_corrente": 60 * 24.0 * 7,
            "cattalini_maregrafo": 60 * 24.0 * 7,
            "cattalini_meteorologia": 60 * 24.0 * 7,
            "odas_corrente": 60 * 24.0 * 7,
            "odas_meteorologia": 60 * 24.0 * 7,
            "porto_astronomica": 60 * 24.0 * 7,
            "porto_harmonico": 60 * 24.0 * 7,
            "porto_maregrafo": 60 * 24.0 * 7,
        },
    )

    target_window_lengths = defaultdict(
        float,
        {
            "cattalini_meteorologia": 60 * 24.0 * 2,
            "odas_corrente": 60 * 24.0 * 2,
            "porto_maregrafo": 60 * 24.0 * 2,
        },
    )

    max_forecast_window_lengths = defaultdict(
        float,
        {
            "cattalini_meteorologia": 60 * 24.0 * 2,
            "odas_corrente": 60 * 24.0 * 2,
            "porto_maregrafo": 60 * 24.0 * 2,
        },
    )

    look_ahead_lengths = defaultdict(
        float,
        {
            "porto_astronomica": 60 * 24.0 * 2,
            "porto_harmonico": 60 * 24.0 * 2,
        },
    )

    train_dataset = SantosTrainDatasetTorch(
        data_path=train_data_path,
        context_window_lengths=context_window_lengths,
        target_window_lengths=target_window_lengths,
        look_ahead_lengths=look_ahead_lengths,
        sample_size_per_epoch=sample_size_per_epoch,
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=SantosTrainDatasetTorch.collate_fn,
    )

    test_dataset = SantosTestDatasetTorch(
        data_path=test_data_path,
        context_masks_path=test_data_path / "missing_ratio_0" / "context_masks",
        target_masks_path=test_data_path / "missing_ratio_0" / "target_masks",
        max_context_size=max_context_window_lengths,
    )

    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SantosTestDatasetTorch.collate_fn,
    )

    model = GapAheadAMTSRegressor(
        input_sizes=train_dataset.n_features,
        num_layers_rnns=model_config.num_layers_rnn,
        hidden_units=model_config.hidden_units,
        time_encoder=(
            PositionalEncoding(time_encoding_size=model_config.time_encoding_size)
            if model_config.time_encoding_size > 0
            else None
        ),
        num_layers_gnn=model_config.num_layers_gnn,
        num_heads_gnn=model_config.num_heads_gnn,
    )

    if start_model_path != "":
        model.load_state_dict(torch.load(start_model_path))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        batch_count = 0
        for (
            ((x_timestamps, x_features), y_timestamps),
            y_features,
        ), t_inferences in train_dl:

            x_timestamps = {
                ts_name: [xi.to(device) for xi in x]
                for ts_name, x in x_timestamps.items()
            }
            x_features = {
                ts_name: [xi.to(device) for xi in x]
                for ts_name, x in x_features.items()
            }
            y_timestamps = {
                ts_name: [yi.to(device) for yi in y]
                for ts_name, y in y_timestamps.items()
            }

            y_features = {
                ts_name: [yi.to(device) for yi in y]
                for ts_name, y in y_features.items()
            }
            t_inferences = t_inferences.to(device)
            model.train()

            model.zero_grad()
            forecast = model(
                context_timestamps=x_timestamps,
                context_features=x_features,
                target_timestamps=y_timestamps,
                t_inferences=t_inferences,
            )
            losses_by_ts = {
                ts_name: torch.stack(
                    [
                        1.0 - index_agreement_torch(f[i], y_features[ts_name][i])
                        for i in range(len(f))
                        if f[i].size(0) > 0
                    ]
                )
                for ts_name, f in forecast.items()
            }

            # losses_by_ts = {
            #     ts_name: torch.stack(
            #         [
            #             torch.nn.MSELoss()(f[i], y_features[ts_name][i]).view(1)
            #             for i in range(len(f))
            #             if f[i].size(0) > 0
            #         ]
            #     )
            #     for ts_name, f in forecast.items()
            # }

            loss = torch.cat(
                [mean.mean(dim=0) for mean in losses_by_ts.values()],
                dim=-1,
            ).mean()
            loss.backward()
            optimizer.step()
            print(
                (
                    f"Iteration: {str(batch_count+1).rjust(3,"0")}/{len(train_dl)}, "
                    f"Forecast Sizes: {list(target_window_lengths.values())}, "
                    + " - ".join(
                        [
                            f"{ts_name}: {[f'({m:.2f},{s:.2f})' for (m,s) in zip(loss.mean(dim=0).tolist(),loss.std(dim=0).tolist())]}"
                            for ts_name, loss in losses_by_ts.items()
                        ]
                    )
                )
            )
            batch_count += 1

        test_results: dict[str, list[torch.Tensor]] = defaultdict(list)
        model.zero_grad()

        batch_count = 0

        for (
            ((x_timestamps, x_features), y_timestamps),
            y_features,
        ) in test_dl:

            x_timestamps = {
                ts_name: [xi.to(device) for xi in x]
                for ts_name, x in x_timestamps.items()
            }
            x_features = {
                ts_name: [xi.to(device) for xi in x]
                for ts_name, x in x_features.items()
            }
            y_timestamps = {
                ts_name: [yi.to(device) for yi in y]
                for ts_name, y in y_timestamps.items()
            }

            y_features = {
                ts_name: [yi.to(device) for yi in y]
                for ts_name, y in y_features.items()
            }
            model.eval()

            with torch.no_grad():
                t_inferences = torch.stack(
                    [
                        torch.stack(
                            [
                                (
                                    timestamp_tensor[0]
                                    if timestamp_tensor.size(0) > 0
                                    else torch.tensor(
                                        torch.inf, device=timestamp_tensor.device
                                    )
                                )
                                for i, timestamp_tensor in enumerate(y)
                            ]
                        )
                        for ts_name, y in y_timestamps.items()
                    ],
                    dim=-1,
                ).min(dim=-1)[0]

                forecast = model(
                    context_timestamps=x_timestamps,
                    context_features=x_features,
                    target_timestamps=y_timestamps,
                    t_inferences=t_inferences,
                )

                losses_by_ts = {
                    ts_name: torch.stack(
                        [
                            1.0 - index_agreement_torch(f[i], y_features[ts_name][i])
                            for i in range(len(f))
                            if f[i].size(0) > 0
                        ]
                    )
                    for ts_name, f in forecast.items()
                }

            for ts_name, loss in losses_by_ts.items():
                test_results[ts_name].append(loss.detach().cpu().mean(dim=0))

            if batch_count == len(test_dl) // 2:
                for ts_name, y_ts_list in y_timestamps.items():
                    if ts_name == "cattalini_meteorologia":
                        feats = ["ws", "wd"]
                        for i, f in enumerate(feats):
                            uniplot.plot(
                                xs=[
                                    y_ts_list[0].cpu().numpy(),
                                    y_ts_list[0].cpu().numpy(),
                                ],
                                ys=[
                                    y_features[ts_name][0][:, i]
                                    .squeeze()
                                    .cpu()
                                    .numpy(),
                                    forecast[ts_name][0][:, i]
                                    .squeeze()
                                    .cpu()
                                    .detach()
                                    .numpy(),
                                ],
                                color=True,
                                legend_labels=["Target", "Forecast"],
                                title=f"{ts_name} {f}",
                            )
                    elif ts_name == "odas_corrente":

                        feats = ["cs", "cd"]
                        for i, f in enumerate(feats):
                            uniplot.plot(
                                xs=[
                                    y_ts_list[0].cpu().numpy(),
                                    y_ts_list[0].cpu().numpy(),
                                ],
                                ys=[
                                    y_features[ts_name][0][:, i]
                                    .squeeze()
                                    .cpu()
                                    .numpy(),
                                    forecast[ts_name][0][:, i]
                                    .squeeze()
                                    .cpu()
                                    .detach()
                                    .numpy(),
                                ],
                                color=True,
                                legend_labels=["Target", "Forecast"],
                                title=f"{ts_name} {f}",
                            )
                    elif ts_name == "porto_maregrafo":

                        feats = ["ssh"]
                        for i, f in enumerate(feats):
                            uniplot.plot(
                                xs=[
                                    y_ts_list[0].cpu().numpy(),
                                    y_ts_list[0].cpu().numpy(),
                                ],
                                ys=[
                                    y_features[ts_name][0][:, i]
                                    .squeeze()
                                    .cpu()
                                    .numpy(),
                                    forecast[ts_name][0][:, i]
                                    .squeeze()
                                    .cpu()
                                    .detach()
                                    .numpy(),
                                ],
                                color=True,
                                legend_labels=["Target", "Forecast"],
                                title=f"{ts_name} {f}",
                            )

            batch_count += 1

        test_results_ = {ts_name: torch.stack(losses).mean(dim=0) for ts_name, losses in test_results.items()}  # type: ignore
        print(f"\n\nTEST RESULTS EPOCH {epoch}: {test_results_}\n\n")
        context_window_lengths = {
            ts_name: min(
                context_window_lengths[ts_name] + context_increase_step,
                max_context_window_lengths[ts_name],
            )
            for ts_name in context_window_lengths
        }
        train_dataset.context_window_lengths = context_window_lengths

        target_window_lengths = {
            ts_name: min(
                target_window_lengths[ts_name] + forecast_increase_step,
                max_forecast_window_lengths[ts_name],
            )
            for ts_name in target_window_lengths
        }
        train_dataset.target_window_lengths = target_window_lengths
        torch.save(model.state_dict(), out_path / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
