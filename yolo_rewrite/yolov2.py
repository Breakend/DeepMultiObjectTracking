import chainer.links as L
from chainer import Chain

from lib.functions import *
from lib.utils import *


class YOLOv2(object):
    """
    YOLOv2
    - It takes (416, 416, 3) sized image as input
    """

class YOLOv2Predictor(Chain):
    def __init__(self, predictor):
        super(YOLOv2Predictor, self).__init__(predictor=predictor)
        self.anchors = predictor.anchors
        self.thresh = 0.6
        self.seen = 0
        self.unstable_seen = 5000

    def __call__(self, input_x, t):
        output = self.predictor(input_x)
        batch_size, _, grid_h, grid_w = output.shape
        self.seen += batch_size
        x, y, w, h, conf, prob = F.split_axis(F.reshape(output, (
            batch_size, self.predictor.n_boxes, self.predictor.n_classes + 5,
            grid_h, grid_w)), (1, 2, 3, 4, 5), axis=2)
        x = F.sigmoid(x)  # xのactivation
        y = F.sigmoid(y)  # yのactivation
        conf = F.sigmoid(conf)  # confのactivation
        prob = F.transpose(prob, (0, 2, 1, 3, 4))
        prob = F.softmax(prob)  # probablitiyのacitivation

        # 教師データの用意
        tw = np.zeros(w.shape,
                      dtype=np.float32)  # wとhが0になるように学習(e^wとe^hは1に近づく -> 担当するbboxの倍率1)
        th = np.zeros(h.shape, dtype=np.float32)
        tx = np.tile(0.5, x.shape).astype(np.float32)  # 活性化後のxとyが0.5になるように学習()
        ty = np.tile(0.5, y.shape).astype(np.float32)

        if self.seen < self.unstable_seen:  # centerの存在しないbbox誤差学習スケールは基本0.1
            box_learning_scale = np.tile(0.1, x.shape).astype(np.float32)
        else:
            box_learning_scale = np.tile(0, x.shape).astype(np.float32)

        tconf = np.zeros(conf.shape,
                         dtype=np.float32)  # confidenceのtruthは基本0、iouがthresh以上のものは学習しない、ただしobjectの存在するgridのbest_boxのみ真のIOUに近づかせる
        conf_learning_scale = np.tile(0.1, conf.shape).astype(np.float32)

        tprob = prob.data.copy()  # best_anchor以外は学習させない(自身との二乗和誤差 = 0)

        # 全bboxとtruthのiouを計算(batch単位で計算する)
        x_shift = Variable(
            np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape[1:]))
        y_shift = Variable(np.broadcast_to(
            np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1),
            y.shape[1:]))
        w_anchor = Variable(np.broadcast_to(
            np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0],
                       (self.predictor.n_boxes, 1, 1, 1)), w.shape[1:]))
        h_anchor = Variable(np.broadcast_to(
            np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1],
                       (self.predictor.n_boxes, 1, 1, 1)), h.shape[1:]))
        x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
        best_ious = []
        for batch in range(batch_size):
            n_truth_boxes = len(t[batch])
            box_x = (x[batch] + x_shift) / grid_w
            box_y = (y[batch] + y_shift) / grid_h
            box_w = F.exp(w[batch]) * w_anchor / grid_w
            box_h = F.exp(h[batch]) * h_anchor / grid_h

            ious = []
            for truth_index in range(n_truth_boxes):
                truth_box_x = Variable(np.broadcast_to(
                    np.array(t[batch][truth_index]["x"], dtype=np.float32),
                    box_x.shape))
                truth_box_y = Variable(np.broadcast_to(
                    np.array(t[batch][truth_index]["y"], dtype=np.float32),
                    box_y.shape))
                truth_box_w = Variable(np.broadcast_to(
                    np.array(t[batch][truth_index]["w"], dtype=np.float32),
                    box_w.shape))
                truth_box_h = Variable(np.broadcast_to(
                    np.array(t[batch][truth_index]["h"], dtype=np.float32),
                    box_h.shape))
                truth_box_x.to_gpu(), truth_box_y.to_gpu(), truth_box_w.to_gpu(), truth_box_h.to_gpu()
                ious.append(multi_box_iou(Box(box_x, box_y, box_w, box_h),
                                          Box(truth_box_x, truth_box_y,
                                              truth_box_w,
                                              truth_box_h)).data.get())
            ious = np.array(ious)
            best_ious.append(np.max(ious,
                                    axis=0))  # max is on gt axis (only one box for each gt)
        best_ious = np.array(best_ious)

        # 一定以上のiouを持つanchorに対しては、confを0に下げないようにする(truthの周りのgridはconfをそのまま維持)。
        tconf[best_ious > self.thresh] = conf.data.get()[
            best_ious > self.thresh]
        conf_learning_scale[best_ious > self.thresh] = 0

        # objectの存在するanchor boxのみ、x、y、w、h、conf、probを個別修正
        abs_anchors = self.anchors / np.array([grid_w, grid_h])
        for batch in range(batch_size):
            for truth_box in t[batch]:
                truth_w = int(float(truth_box["x"]) * grid_w)
                truth_h = int(float(truth_box["y"]) * grid_h)
                truth_n = 0
                best_iou = 0.0
                for anchor_index, abs_anchor in enumerate(abs_anchors):
                    iou = box_iou(Box(0, 0, float(truth_box["w"]),
                                      float(truth_box["h"])),
                                  Box(0, 0, abs_anchor[0], abs_anchor[1]))
                    if best_iou < iou:
                        best_iou = iou
                        truth_n = anchor_index

                # objectの存在するanchorについて、centerを0.5ではなく、真の座標に近づかせる。anchorのスケールを1ではなく真のスケールに近づかせる。学習スケールを1にする。
                box_learning_scale[batch, truth_n, :, truth_h, truth_w] = 1.0
                tx[batch, truth_n, :, truth_h, truth_w] = float(
                    truth_box["x"]) * grid_w - truth_w
                ty[batch, truth_n, :, truth_h, truth_w] = float(
                    truth_box["y"]) * grid_h - truth_h
                tw[batch, truth_n, :, truth_h, truth_w] = np.log(
                    float(truth_box["w"]) / abs_anchors[truth_n][0])
                th[batch, truth_n, :, truth_h, truth_w] = np.log(
                    float(truth_box["h"]) / abs_anchors[truth_n][1])
                tprob[batch, :, truth_n, truth_h, truth_w] = 0
                tprob[batch, int(
                    truth_box["label"]), truth_n, truth_h, truth_w] = 1

                # IOUの観測
                full_truth_box = Box(float(truth_box["x"]),
                                     float(truth_box["y"]),
                                     float(truth_box["w"]),
                                     float(truth_box["h"]))
                predicted_box = Box(
                    (x[batch][truth_n][0][truth_h][
                         truth_w].data.get() + truth_w) / grid_w,
                    (y[batch][truth_n][0][truth_h][
                         truth_w].data.get() + truth_h) / grid_h,
                    np.exp(w[batch][truth_n][0][truth_h][truth_w].data.get()) *
                    abs_anchors[truth_n][0],
                    np.exp(h[batch][truth_n][0][truth_h][truth_w].data.get()) *
                    abs_anchors[truth_n][1]
                )
                predicted_iou = box_iou(full_truth_box, predicted_box)
                tconf[batch, truth_n, :, truth_h, truth_w] = predicted_iou
                conf_learning_scale[batch, truth_n, :, truth_h, truth_w] = 10.0

            # debug prints
            # maps = F.transpose(prob[batch], (2, 3, 1, 0)).data
            # print("best confidences and best conditional probability and predicted class of each grid:")
            # for i in range(grid_h):
            #     for j in range(grid_w):
            #         print("%2d" % (int(conf[batch, :, :, i, j].data.max() * 100)), end=" ")
            #     print("     ", end="")
            #     for j in range(grid_w):
            #         print("%2d" % (maps[i][j][int(maps[i][j].max(axis=1).argmax())].argmax()), end=" ")
            #     print("     ", end="")
            #     for j in range(grid_w):
            #         print("%2d" % (maps[i][j][int(maps[i][j].max(axis=1).argmax())].max()*100), end=" ")
            #     print()

            print(
                "best default iou: %.2f   predicted iou: %.2f   confidence: %.2f   class: %s" % (
                    best_iou, predicted_iou,
                    conf[batch][truth_n][0][truth_h][truth_w].data,
                    t[batch][0]["label"]))
            print("-------------------------------")
        print("seen = %d" % self.seen)

        # loss計算
        tx, ty, tw, th, tconf, tprob = Variable(tx), Variable(ty), Variable(
            tw), Variable(th), Variable(tconf), Variable(tprob)
        box_learning_scale, conf_learning_scale = Variable(
            box_learning_scale), Variable(conf_learning_scale)
        tx.to_gpu(), ty.to_gpu(), tw.to_gpu(), th.to_gpu(), tconf.to_gpu(), tprob.to_gpu()
        box_learning_scale.to_gpu()
        conf_learning_scale.to_gpu()

        # print(tw.data[0], w.data[0])
        x_loss = F.sum((tx - x) ** 2 * box_learning_scale) / 2
        y_loss = F.sum((ty - y) ** 2 * box_learning_scale) / 2
        w_loss = F.sum((tw - w) ** 2 * box_learning_scale) / 2
        h_loss = F.sum((th - h) ** 2 * box_learning_scale) / 2
        c_loss = F.sum((tconf - conf) ** 2 * conf_learning_scale) / 2
        p_loss = F.sum((tprob - prob) ** 2) / 2
        print(
            "x_loss: %f  y_loss: %f  w_loss: %f  h_loss: %f  c_loss: %f   p_loss: %f" %
            (F.sum(x_loss).data, F.sum(y_loss).data, F.sum(w_loss).data,
             F.sum(h_loss).data, F.sum(c_loss).data, F.sum(p_loss).data)
        )

        loss = x_loss + y_loss + w_loss + h_loss + c_loss + p_loss
        return loss

    def init_anchor(self, anchors):
        self.anchors = anchors

    def predict(self, input_x):
        output = self.predictor(input_x)
        batch_size, _, grid_h, grid_w = output.shape
        x, y, w, h, conf, prob = F.split_axis(F.reshape(output, (
            batch_size, self.predictor.n_boxes, self.predictor.n_classes + 5,
            grid_h, grid_w)), (1, 2, 3, 4, 5), axis=2)
        x = F.sigmoid(x)
        y = F.sigmoid(y)
        conf = F.sigmoid(conf)
        prob = F.transpose(prob, (0, 2, 1, 3, 4))
        prob = F.softmax(prob)  # softmax is performed on channel (= 2nd axis)
        prob = F.transpose(prob, (0, 2, 1, 3, 4))

        # x, y, w, hを絶対座標へ変換
        x_shift = Variable(
            np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape),
            volatile=True)
        y_shift = Variable(np.broadcast_to(
            np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape),
            volatile=True)
        w_anchor = Variable(np.broadcast_to(
            np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0],
                       (self.predictor.n_boxes, 1, 1, 1)), w.shape),
            volatile=True)
        h_anchor = Variable(np.broadcast_to(
            np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1],
                       (self.predictor.n_boxes, 1, 1, 1)), h.shape),
            volatile=True)
        x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
        box_x = (x + x_shift) / grid_w
        box_y = (y + y_shift) / grid_h
        box_w = F.exp(w) * w_anchor / grid_w
        box_h = F.exp(h) * h_anchor / grid_h
        box_left = box_x - box_w / 2
        box_top = box_y - box_h / 2
        box_right = box_x + box_w / 2
        box_bottom = box_y + box_h / 2
        boxes = F.concat((box_left, box_top, box_right, box_bottom), axis=2)
        boxes = F.transpose(boxes, (0, 1, 3, 4, 2))
        conf = F.transpose(conf, (0, 1, 3, 4, 2))
        prob = F.transpose(prob, (0, 1, 3, 4, 2))
        return boxes, conf, prob
