import os
import numpy as np
from GradientExtraction.Radial.hyperparam import RESULTS


def rgb2hex(rgb):
    if 0<= rgb[0] <=1  and 0<= rgb[1] <=1 and 0<= rgb[2] <=1:
        rgb = np.clip(rgb*255, 0, 255)
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def create_linear(size, stops, colors, name, path):
    assert len(size) == 2 and len(stops) == len(colors) and len(stops) > 1 and colors.shape[1] == 3
    percentages = [0] + [np.linalg.norm(stops[i] - stops[i - 1]) for i in range(1, len(stops))]
    percentages = np.cumsum(percentages)
    percentages = percentages / percentages[-1]

    linearGradient = '<linearGradient id="linear-gradient" x1="{}" y1="{}" x2="{}" y2="{}" gradientUnits="userSpaceOnUse">'.format(
        stops[0, 1], stops[0, 0], stops[-1, 1], stops[-1, 0])
    stops = ''''''
    for i, p in enumerate(percentages):
        stops += '\t\t\t<stop offset="{:.3f}" style="stop-color:rgb({}, {}, {});stop-opacity:1" />\n'.format(p, colors[i, 0],
                                                                                                       colors[i, 1],
                                                                                                       colors[i, 2])

    header = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {} {}">'.format(
        size[1], size[0])
    rect = '<rect class="cls-1" x="0.0" y="0.0" width="{}" height="{}"/>'.format(size[1], size[0])
    stroke = '{stroke:#000;stroke-miterlimit:10;fill:url(#linear-gradient);}'
    svg = "{}\n" \
          "\t<defs>\n" \
          "\t\t<style>.cls-1{}</style>\n" \
          "\t\t{}\n" \
          "{}" \
          "\t\t</linearGradient>\n" \
          "\t</defs>\n" \
          "\t<g id=\"Layer_2\" data-name=\"Layer 2\"><g id=\"Layer_1-2\" data-name=\"Layer 1\">" \
          "\t{}\n" \
          "\t</g></g>\n" \
          "</svg>".format(header,stroke, linearGradient, stops, rect)

    # print(svg)
    with open(os.path.join(RESULTS, name+'_linear.svg'), 'w') as file:
        file.write(svg)


def create_concentric_radial(size, center, radius, stop_percentage, stop_colors, name=""):

    header = '<?xml version="1.0" encoding="UTF-8"?>\n' \
             '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {} {}">'.format(size[1], size[0])
    stroke = '.e{fill:url(#d);}'

    radial_gradient = '<radialGradient id="d" cx="{}" cy="{}" fx="{}" fy="{}" r="{}" ' \
                      'gradientUnits="userSpaceOnUse">'.format(center[0], center[1], center[0], center[1], radius)
    stops = ''''''
    for i, p in enumerate(stop_percentage):
        stops += '\t\t\t<stop offset="{:.3f}" stop-color="{}"/>\n'.format(p, rgb2hex(stop_colors[i]))

    svg = "{}\n" \
          "\t<defs>\n" \
          "\t\t<style>{}</style>\n" \
          "\t\t{}\n" \
          "{}" \
          "\t\t</radialGradient>\n" \
          "\t</defs>\n" \
          "\t<g id=\"a\"/>\n" \
          "\t<g id=\"b\">\n" \
          "\t\t<g id=\"c\">\n" \
          "\t\t\t<rect class=\"e\" width=\"{}\" height=\"{}\"/>\n" \
          "\t\t</g>\n" \
          "\t</g>\n" \
          "</svg>".format(header, stroke, radial_gradient, stops, size[1], size[0])

    # print(svg)
    with open(os.path.join(RESULTS, name+'_concentric.svg'), 'w') as file:
        file.write(svg)


def create_eccentric_radial(size, focal, center, radius, stop_percentage, stop_colors, name=""):

    header = '<?xml version="1.0" encoding="UTF-8"?>\n' \
             '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {} {}">'.format(size[1], size[0])
    stroke = '.e{fill:url(#d);}'

    radial_gradient = '<radialGradient id="d" cx="{}" cy="{}" fx="{}" fy="{}" r="{}" ' \
                      'gradientUnits="userSpaceOnUse">'.format(center[0], center[1], focal[0], focal[1], radius)
    stops = ''''''
    for i, p in enumerate(stop_percentage):
        stops += '\t\t\t<stop offset="{:.3f}" stop-color="{}"/>\n'.format(p, rgb2hex(stop_colors[i]))

    svg = "{}\n" \
          "\t<defs>\n" \
          "\t\t<style>{}</style>\n" \
          "\t\t{}\n" \
          "{}" \
          "\t\t</radialGradient>\n" \
          "\t</defs>\n" \
          "\t<g id=\"a\"/>\n" \
          "\t<g id=\"b\">\n" \
          "\t\t<g id=\"c\">\n" \
          "\t\t\t<rect class=\"e\" width=\"{}\" height=\"{}\"/>\n" \
          "\t\t</g>\n" \
          "\t</g>\n" \
          "</svg>".format(header, stroke, radial_gradient, stops, size[1], size[0])

    # print(svg)
    with open(os.path.join(RESULTS, name+'_eccentric.svg'), 'w') as file:
        file.write(svg)


def create_ellipse(size, stops, colors, focal, outer_center, outer_radius, scale, rotation):
    cx, cy = outer_center[0], outer_center[1]
    fx, fy = focal[0], focal[1]
    stop_length = np.linalg.norm(stops - focal, axis=1)
    stop_percentage = stop_length / stop_length[-1]

    center = np.array(size) * 0.5
    tx, ty = center[1] - cx, (size[0] - (center[0]-cy))


    header = '<?xml version="1.0" encoding="UTF-8"?>\n' \
             '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {} {}">'.format(
        size[1], size[0])
    rect = '<rect class="e" width="{}" height="{}"/>'.format(size[1], size[0])
    stroke = '.e{fill:url(#d);}'

    radial_gradient = '<radialGradient id="d" cx="{}" cy="{}" fx="{}" fy="{}" r="{}" ' \
                      'gradientTransform="translate({} {}) rotate({}) scale(1 {})" ' \
                      'gradientUnits="userSpaceOnUse">'.format(center[1], center[0], fx, fy, outer_radius, tx, ty, np.rad2deg(rotation), scale)
    stops = ''''''
    for i, p in enumerate(stop_percentage):
        stops += '\t\t\t<stop offset="{:.3f}" stop-color="{}"/>\n'.format(p, rgb2hex(colors[i]))

    svg = "{}\n" \
          "\t<defs>\n" \
          "\t\t<style>{}</style>\n" \
          "\t\t{}\n" \
          "{}" \
          "\t\t</radialGradient>" \
          "</defs>\n" \
          "\t<g id=\"a\"/>\n" \
          "\t<g id=\"b\"> <g id=\"c\">" \
          "\t\t{}\n" \
          "\t\t</g>" \
          "\t</g>\n" \
          "</svg>".format(header,stroke, radial_gradient, stops, rect)

    print(svg)
    with open(os.path.join(RESULTS, 'ellipse.svg'), 'w') as file:
        file.write(svg)


def create_dummy_color_radial_grad(cols_rows, origin, focal, outer_radius, T):
    header = '<svg viewBox="0 0 {}, {}" xmlns="http://www.w3.org/2000/svg">'.format(cols_rows[0], cols_rows[1])
    fo = focal - origin
    radial_gradient_begin = '\t<radialGradient id="gradient1" gradientUnits="userSpaceOnUse" \n\t\t' \
                      'cx="0" cy="0"  fx="{}" fy="{}" r="{}"\n\t\t' \
                      'gradientTransform="matrix({}, {}, {}, {}, {}, {})">'.format(fo[0], fo[1], outer_radius,
                                                                                   T[0,0], T[0,1], T[1,0], T[1,1], origin[0], origin[1])
    stops = '\t<stop offset="0%" stop-color="darkblue" /> ' \
            '\n\t<stop offset="20%" stop-color="skyblue" /> ' \
            '\n\t<stop offset="21%" stop-color="darkblue" />' \
            '\n\t<stop offset="22%" stop-color="skyblue" />' \
            '\n\t<stop offset="50%" stop-color="skyblue" />' \
            '\n\t<stop offset="51%" stop-color="darkblue" />' \
            '\n\t<stop offset="52%" stop-color="skyblue" />' \
            '\n\t<stop offset="80%" stop-color="darkblue" />' \
            '\n\t<stop offset="81%" stop-color="skyblue" />' \
            '\n\t<stop offset="82%" stop-color="darkblue" />' \
            '\n\t<stop offset="100%" stop-color="darkblue" />'
    radial_gradient_end = '\t</radialGradient>'
    rect = '\t<rect x="0" y="0" width="{}" height="{}" fill="url(#gradient1)" />'.format(cols_rows[0], cols_rows[1])
    end = '</svg>'

    svg = '\n'.join([header, radial_gradient_begin, stops, radial_gradient_end, rect, end])
    print("----------------------------------")
    print(svg)
    print("----------------------------------")
    with open(os.path.join(RESULTS, 'fina1.svg'), 'w') as file:
        file.write(svg)


def create_radial_grad_ecc_T(cols_rows, oHat, fHat, outer_center, outer_radius, T, stops):
    """
    :param cols_rows: width, height
    :param ecc: Eccentricity of the skewed centers
    :param focal: focal
    :param outer_radius: Outer radius
    :param T: Transform
    :param stops: tuple of list of color stops and distance
    :return:
    """
    header = '<svg viewBox="0 0 {}, {}" xmlns="http://www.w3.org/2000/svg">'.format(cols_rows[0], cols_rows[1])
    fo = fHat - oHat
    co = outer_center - oHat
    radial_gradient_begin = '\t<radialGradient id="gradient1" gradientUnits="userSpaceOnUse" \n\t\t' \
                            'cx="{}" cy="{}"  fx="{}" fy="{}" r="{}"\n\t\t' \
                            'gradientTransform="matrix({}, {}, {}, {}, {}, {})">'.format(co[0], co[1],
                                                                                         fo[0], fo[1], outer_radius,
                                                                                         T[0, 0], T[0, 1],
                                                                                         T[1, 0], T[1, 1],
                                                                                         oHat[0], oHat[1])
    stops_str = []
    for c, p in zip(*stops):
        stops_str.append('\t<stop offset="{}%" style="stop-color:{}" />'.format(p*100, rgb2hex(c)))

    radial_gradient_end = '\t</radialGradient>'
    rect = '\t<rect x="0" y="0" width="{}" height="{}" fill="url(#gradient1)" />'.format(cols_rows[0], cols_rows[1])
    end = '</svg>'

    svg = '\n'.join([header, radial_gradient_begin, *stops_str, radial_gradient_end, rect, end])
    print("----------------------------------")
    print(svg)
    print("----------------------------------")
    os.makedirs(os.path.join(RESULTS), exist_ok=True)
    with open(os.path.join(RESULTS, 'dump3.svg'), 'w') as file:
        file.write(svg)



if __name__ == '__main__':
    create_ellipse(size=(64, 64), stops=np.array([[32, 43], [34, 24], [36, 7]]),
                   colors=np.array([[254, 253, 241], [245, 238, 51], [237, 33, 124]]), focal=np.array([32, 43]),
                   outer_center=np.array([36, 7]), outer_radius=36, scale=0.64, rotation=-45)