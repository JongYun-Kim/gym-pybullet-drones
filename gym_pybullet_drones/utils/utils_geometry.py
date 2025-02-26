import numpy as np


def project_point_on_pad_plane(pt_3d, pad_center_3d, R_pad):
    """
    pt_3d를 패드 평면의 '로컬 2D 좌표' (x,y)에 매핑.
    여기서 패드 로컬 x축 = R_pad * (1,0,0),
                     y축 = R_pad * (0,1,0) 라고 가정.
    """
    # 로컬 축 추출
    pad_x_axis = R_pad[:, 0]  # 첫 번째 컬럼
    pad_y_axis = R_pad[:, 1]  # 두 번째 컬럼
    # 패드 평면에서 pt_3d까지의 벡터
    v = pt_3d - pad_center_3d
    x_local = np.dot(v, pad_x_axis)
    y_local = np.dot(v, pad_y_axis)
    return np.array([x_local, y_local])


def inside_pad_box(pt_2d, half_size=0.25):
    """
    pt_2d가 0.5×0.5 정사각형(센터[0,0]) 안에 있으면 True
    """
    x, y = pt_2d
    if -half_size <= x <= half_size and -half_size <= y <= half_size:
        return True
    return False


def order_points_convex_polygon(points_2d):
    """
    2D 점들(points_2d)이 볼록다각형이라 가정하고,
    시계방향(또는 반시계방향) 순서로 정렬해 반환.
    간단히 '평균점 대비 각도'로 소팅 가능.
    """
    pts = np.array(points_2d)
    center = pts.mean(axis=0)
    vecs = pts - center
    angles = np.arctan2(vecs[:,1], vecs[:,0])
    sort_idx = np.argsort(angles)
    return pts[sort_idx]


def polygons_intersect_2d(polyA, polyB):
    """
    볼록다각형 polyA, polyB 가 2D 평면에서 교차하는지
    Separating Axis Theorem(SAT)을 이용해 판단.
    polyA, polyB는 shape=(N,2), shape=(M,2).
    """
    # 양쪽 다각형에 대해 모든 에지의 법선을 검사
    #   1) 에지를 (x2 - x1, y2 - y1)로 구하고
    #   2) 그에 수직인 축(normal)을 만들어 투영
    #   3) 두 다각형의 투영 구간이 겹치지 않으면 분리됨 → 교차X
    #   4) 모든 에지에 대해 겹치면 교차O

    def edges_and_normals(poly):
        edges = []
        normals = []
        for i in range(len(poly)):
            j = (i+1) % len(poly)
            edge = poly[j] - poly[i]  # (dx, dy)
            # 에지에 수직인 벡터 예: (dy, -dx)
            normal = np.array([edge[1], -edge[0]])
            # 정규화(optional)
            nlen = np.linalg.norm(normal)
            if nlen < 1e-12:
                continue
            normal /= nlen
            edges.append(edge)
            normals.append(normal)
        return edges, normals

    def project_polygon(poly, axis):
        # poly의 모든 점을 axis에 내적하여 min, max 구함
        dots = poly @ axis
        return np.min(dots), np.max(dots)

    def overlap_1d(a_min, a_max, b_min, b_max):
        return not (a_max < b_min or b_max < a_min)

    # 두 볼록다각형에 대해 모든 에지→법선 추출
    edgesA, normalsA = edges_and_normals(polyA)
    edgesB, normalsB = edges_and_normals(polyB)

    # 모든 법선에 대해(즉, polyA의 에지 법선 + polyB의 에지 법선)
    for n in (normalsA + normalsB):
        # 각 폴리곤을 n축에 투영
        a_min, a_max = project_polygon(polyA, n)
        b_min, b_max = project_polygon(polyB, n)
        # 겹치지 않으면 교차X
        if not overlap_1d(a_min, a_max, b_min, b_max):
            return False

    # 끝까지 분리축이 없으면 교차
    return True

def line_plane_intersection(line_point, line_dir, plane_point, plane_normal, eps=1e-7):
    """
    line_point + t * line_dir 이 plane_point, plane_normal 인 평면과 만나는 t를 구함.
    """
    denom = np.dot(plane_normal, line_dir)
    if abs(denom) < eps:
        return None  # 평행 or 거의 평행
    t = np.dot(plane_normal, (plane_point - line_point)) / denom
    # 교차점이 카메라 뒤(t<0)인지, 앞(t>0)인지도 체크할 수 있음
    if t <= 0:
        return None  # 카메라 '앞'이 아니라 뒤쪽에 교차 or 평면이 뒤쪽에 있음
    return line_point + t * line_dir