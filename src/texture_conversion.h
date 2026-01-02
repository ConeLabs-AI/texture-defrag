#ifndef TEXTURE_CONVERSION_H
#define TEXTURE_CONVERSION_H

#include <string>

namespace TextureConversion {

const int TILE_SIZE = 256;

/**
 * Converts a standard image file (JPEG, PNG, etc.) to the .rawtile format.
 * The .rawtile format consists of:
 * - Header: width (int32), height (int32), tile_size (int32)
 * - Data: Sequence of uncompressed RGBA blocks of tile_size x tile_size pixels.
 * Tiles are stored in row-major order.
 */
bool ConvertToRawTile(const std::string& inputPath, const std::string& outputPath);

std::string GetRawTilePath(const std::string& imagePath);

} // namespace TextureConversion

#endif // TEXTURE_CONVERSION_H