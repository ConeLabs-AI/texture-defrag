#include "texture_conversion.h"
#include "logging.h"
#include <QImage>
#include <QImageReader>
#include <fstream>
#include <vector>
#include <cstdint>

namespace TextureConversion {

bool ConvertToRawTile(const std::string& inputPath, const std::string& outputPath) {
    QImageReader reader(inputPath.c_str());
    if (!reader.canRead()) {
        LOG_ERR << "Failed to read input image for conversion: " << inputPath;
        return false;
    }

    QImage img = reader.read();
    if (img.isNull()) {
        LOG_ERR << "Loaded image is null: " << inputPath;
        return false;
    }

    // Convert to RGBA8888 for consistent raw storage
    if (img.format() != QImage::Format_RGBA8888) {
        img = img.convertToFormat(QImage::Format_RGBA8888);
    }

    int width = img.width();
    int height = img.height();

    std::ofstream out(outputPath, std::ios::binary);
    if (!out) {
        LOG_ERR << "Failed to open output file for writing: " << outputPath;
        return false;
    }

    // Write header
    out.write(reinterpret_cast<const char*>(&width), sizeof(int));
    out.write(reinterpret_cast<const char*>(&height), sizeof(int));
    out.write(reinterpret_cast<const char*>(&TILE_SIZE), sizeof(int));

    int tilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    int tilesY = (height + TILE_SIZE - 1) / TILE_SIZE;

    std::vector<uint32_t> tileData(TILE_SIZE * TILE_SIZE, 0); // Transparent black padding

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            // Fill tileData with pixels from img
            std::fill(tileData.begin(), tileData.end(), 0);
            
            int startX = tx * TILE_SIZE;
            int startY = ty * TILE_SIZE;
            
            for (int y = 0; y < TILE_SIZE; ++y) {
                int imgY = startY + y;
                if (imgY >= height) break;
                
                const uint32_t* scanline = reinterpret_cast<const uint32_t*>(img.constScanLine(imgY));
                for (int x = 0; x < TILE_SIZE; ++x) {
                    int imgX = startX + x;
                    if (imgX >= width) break;
                    tileData[y * TILE_SIZE + x] = scanline[imgX];
                }
            }
            
            out.write(reinterpret_cast<const char*>(tileData.data()), tileData.size() * sizeof(uint32_t));
        }
    }

    return true;
}

std::string GetRawTilePath(const std::string& imagePath) {
    return imagePath + ".rawtile";
}

} // namespace TextureConversion

