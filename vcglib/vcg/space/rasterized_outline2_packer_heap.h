/****************************************************************************
* VCGLib                                                            o o     *
* Visual and Computer Graphics Library                            o     o   *
*                                                                _   O  _   *
* Copyright(C) 2004-2016                                           \/)\/    *
* Visual Computing Lab                                            /\/|      *
* ISTI - Italian National Research Council                           |      *
*                                                                    \      *
* All rights reserved.                                                      *
*                                                                           *
* This program is free software; you can redistribute it and/or modify      *   
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation; either version 2 of the License, or         *
* (at your option) any later version.                                       *
*                                                                           *
* This program is distributed in the hope that it will be useful,           *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                         *
*                                                                           *
****************************************************************************/

#ifndef __RASTERIZED_OUTLINE2_PACKER_H__
#define __RASTERIZED_OUTLINE2_PACKER_H__

#include <vcg/space/rect_packer.h>
#include <vcg/complex/algorithms/outline_support.h>
#include <vcg/math/eigen.h>
#include <omp.h>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#pragma intrinsic(_BitScanForward64)
static inline int vcg_ctzll(unsigned __int64 mask) {
    unsigned long index;
    _BitScanForward64(&index, mask);
    return (int)index;
}
#else
static inline int vcg_ctzll(unsigned long long mask) {
    return __builtin_ctzll(mask);
}
#endif

namespace vcg
{

class BitGrid {
public:
    std::vector<uint64_t> data;
    int w = 0;
    int h = 0;

    void init(int width, int height) {
        w = width;
        h = height;
        data.assign((size_t(width) * height + 63) / 64, 0);
    }

    inline void set(int x, int y) {
        int bitIdx = y * w + x;
        data[bitIdx >> 6] |= (1ULL << (bitIdx & 63));
    }

    inline bool get(int x, int y) const {
        int bitIdx = y * w + x;
        return (data[bitIdx >> 6] >> (bitIdx & 63)) & 1ULL;
    }

    void clear() {
        data.clear();
        w = 0;
        h = 0;
    }

    bool empty() const {
        return data.empty();
    }
};

struct InnerGap {
    int pos;
    int extent;

    bool operator<(const InnerGap& other) { return extent < other.extent || (extent == other.extent && pos > other.pos); }
};




class RasterizedOutline2
{
private:
    //the grid is the "bounding grid" of the polygon, which is returned by the rasterization process
    //this is a vector of "bounding grids", there is one for each rasterization (different rotation or whatever)
    std::vector<BitGrid> grids;

    std::vector<int> gw;
    std::vector<int> gh;

    //points: the points which make the polygon
    std::vector<Point2f> points;

    //deltaY: a vector containing the number of cells (for the i-th column starting from left) from the
    //FIRST NON-EMTPY cell at the bottom to the LAST NON-EMPTY CELL at the top (there is one top vector for each rasterization)
    std::vector< std::vector<int> > deltaY;

    //bottom: a vector containing the number of EMPTY cells found starting from the bottom
    //until the first NON-EMPTY cell is found (there is one bottom vector for each rasterization)
    std::vector< std::vector<int> > bottom;

    //deltaX: a vector containing the number of cells (for the i-th row starting from bottom) from the
    //FIRST NON-EMTPY cell at the left to the LAST NON-EMPTY CELL at the right (there is one right vector for each rasterization)
    std::vector< std::vector<int> > deltaX;

    //left: a vector containing the number of EMPTY cells found starting from the left (at the i-th row starting from the bottom)
    //until the first NON-EMPTY cell is found (there is one left vector for each rasterization)
    std::vector< std::vector<int> > left;

    //the area, measured in cells, of the discrete representations of the polygons
    std::vector<int> discreteAreas;

public:
    RasterizedOutline2() { }
    int gridHeight(int i) { return gh.at(i); }
    int gridWidth( int i) { return gw.at(i); }

    std::vector<Point2f>&  getPoints()           { return points; }
    const std::vector<Point2f>&  getPointsConst() const{ return points; }
    BitGrid& getGrids(int rast_i)  { return grids[rast_i]; }

    //get top/bottom/left/right vectors of the i-th rasterization
    std::vector<int>& getDeltaY(int i) { return deltaY[i]; }
    std::vector<int>& getBottom(int i) { return bottom[i]; }
    std::vector<int>& getDeltaX(int i) { return deltaX[i]; }
    std::vector<int>& getLeft(int i) { return left[i]; }
    int& getDiscreteArea(int i) { return discreteAreas[i]; }
    void addPoint(const Point2f& newpoint) { points.push_back(newpoint); }
    void setPoints(const std::vector<Point2f>& newpoints) { points = newpoints; }

    //resets the state of the poly and resizes all the states vectors
    void resetState(int totalRasterizationsNum) {
        discreteAreas.clear();
        deltaY.clear();
        bottom.clear();
        deltaX.clear();
        left.clear();
        grids.clear();

        discreteAreas.resize(totalRasterizationsNum);
        deltaY.resize(totalRasterizationsNum);
        bottom.resize(totalRasterizationsNum);
        deltaX.resize(totalRasterizationsNum);
        left.resize(totalRasterizationsNum);
        grids.resize(totalRasterizationsNum);
        gw.resize(totalRasterizationsNum);
        gh.resize(totalRasterizationsNum);
    }

    void initFromGrid(int rast_i) {
        BitGrid& tetrisGrid = grids[rast_i];

        bool empty = tetrisGrid.empty();
        if (!empty) {
            empty = true;
            for (uint64_t val : tetrisGrid.data) {
                if (val != 0) {
                    empty = false;
                    break;
                }
            }
        }

        if (empty) {
            gw[rast_i] = 0;
            gh[rast_i] = 0;
            discreteAreas[rast_i] = 0;
            tetrisGrid.clear();
            return;
        }

        int gridWidth = tetrisGrid.w;
        int gridHeight = tetrisGrid.h;

        gw[rast_i] = gridWidth;
        gh[rast_i] = gridHeight;

        //compute bottom,
        //where bottom[i] = empty cells from the bottom in the column i
        for (int col = 0; col < gridWidth; col++) {
            int bottom_i = 0;
            for (int row = gridHeight - 1; row >= 0; row--) {
                if (!tetrisGrid.get(col, row)) {
                    bottom_i++;
                }
                else {
                    bottom[rast_i].push_back(bottom_i);
                    break;
                }
            }
        }
        if (bottom[rast_i].size() == 0) assert("ERROR: EMPTY BOTTOM VECTOR"==0);

        //compute top
        //IT ASSUMES THAT THERE IS AT LEAST ONE NON-0 ELEMENT (which should always be the case, even if the poly is just a point)
        //deltaY[i] = for the column i, it stores the number of cells which are between the bottom and the top side of the poly
        for (int col = 0; col < gridWidth; col++) {
            int deltay_i = gridHeight - bottom[rast_i][col];
            for (int row = 0; row < gridHeight; row++) {
                if (!tetrisGrid.get(col, row)) {
                    deltay_i--;
                }
                else {
                    break;
                }
            }
            deltaY[rast_i].push_back(deltay_i);
        }
        if (deltaY[rast_i].size() == 0) assert("ERROR: EMPTY deltaY VECTOR"==0);

        //same meaning as bottom, but for the left side
        //we want left/right sides vector to be ordered so that index 0 is at poly's bottom
        int left_i;
        for (int row = gridHeight-1; row >= 0; --row) {
            //for (int row = 0; row < gridHeight; ++row) {
            left_i = 0;
            for (int col = 0; col < gridWidth; col++) {
                if (!tetrisGrid.get(col, row)) ++left_i;
                else {
                    left[rast_i].push_back(left_i);
                    break;
                }
            }
        }
        if (left[rast_i].size() == 0) assert("ERROR: EMPTY leftSide VECTOR"==0);

        //we want left/right sides vector to be ordered so that index 0 is at poly's bottom
        int deltax_i;
        for (int row = gridHeight-1; row >= 0; --row) {
            //for (int row = 0; row < gridHeight; ++row) {
            deltax_i = gridWidth - left[rast_i][gridHeight - 1 - row];
            for (int col = gridWidth - 1; col >= 0; --col) {
                if (!tetrisGrid.get(col, row)) --deltax_i;
                else {
                    break;
                }
            }
            deltaX[rast_i].push_back(deltax_i);
        }
        if (deltaX[rast_i].size() == 0) assert("ERROR: EMPTY rightSide VECTOR"==0);

        //compute the discreteArea: IT IS THE AREA (measured in grid cells) BETWEEN THE TOP AND BOTTOM SIDES...
        int discreteArea = 0;
        for (size_t i = 0; i < deltaY[rast_i].size(); i++) {
            discreteArea += deltaY[rast_i][i];
        }
        discreteAreas[rast_i] = discreteArea;

        tetrisGrid.clear();
    }
};

template <class ScalarType>
class ComparisonFunctor
{
    typedef std::vector<vcg::Point2<ScalarType>> Outline2Type;

public:
    const std::vector<Outline2Type> & v;
    inline ComparisonFunctor(const std::vector<Outline2Type> & nv ) : v(nv) { }

    inline bool operator() ( int a, int b )
    {
        float area1 = tri::OutlineUtil<ScalarType>::Outline2Area(v[a]);
        float area2 = tri::OutlineUtil<ScalarType>::Outline2Area(v[b]);

        return area1 > area2;
    }
};

template <class SCALAR_TYPE, class RASTERIZER_TYPE>
class RasterizedOutline2Packer
{
    typedef typename vcg::Box2<SCALAR_TYPE> Box2x;
    typedef typename vcg::Point2<SCALAR_TYPE> Point2x;
    typedef typename vcg::Similarity2<SCALAR_TYPE> Similarity2x;

    static constexpr int INVALID_POSITION = -1;

    struct PlacementResult {
        int cost = INT_MAX;
        int rastIndex = -1;
        int polyX = -1;
        int polyY = -1;
        int container = -1;
        bool placedUsingSecondaryHorizon = false;
    };

public:

  class Parameters
  {
  public:

      // The cost function used by the greedy algorithm when evaluating the next best move
      // MinWastedSpace  Chooses the placement that minimizes the wasted space. The wasted
      //                 space is defined as the area difference between the horizon after
      //                 and and before placing the polygon, MINUS the polygon area.
      // LowestHorizon   Chooses the placement that minimizes the maximum horizon increase
      // MixedCost       Left for compatibility reasons. This should behave similarly to
      //                 MinWastedSpace, while also penalizing placements using one horizon
      //                 that result in too much wasted space relative to the other horizon.
      enum CostFuncEnum {
          MinWastedSpace,
          LowestHorizon,
          MixedCost
      };

      CostFuncEnum costFunction;

      // if true, the packing algorithm evaluates polygon 'drops' from both
      // principal directions
      bool doubleHorizon;

      // if true, the packing algorithm keeps track of a secondary horizon used
      // to place polygons in between previously placed ones
      bool innerHorizon;

      // if true, the packing algorithms tries a small number of random
      // permutations of the polygon sequence. This can result in a higher
      // packing efficiency, but increases the running time of the algorithm
      // proportionally to the number of permutations tested
      bool permutations;

      //the number of rasterizations to create for each polygon; It must be a multiple of 4.
      int rotationNum;

      //the width (in pixels) of the gutter added around the outline
      int gutterWidth;

      // if false, then do not combine the costs when doubeHorizon is used. This
      // can help to keep the packing area in a rectangular region
      bool minmax;

      ///default constructor
      Parameters()
      {
          costFunction = LowestHorizon;
          doubleHorizon=true;
          innerHorizon=false;
          permutations=false;
          rotationNum = 16;
          gutterWidth = 0;
          minmax = false;
      }
  };

  //THE CLASS WHICH HANDLES THE PACKING AND THE UPDATED STATE OF THE PACKING ALGORITHMS
  class packingfield
  {

  private:

      using CostFuncEnum = typename Parameters::CostFuncEnum;
      //the bottomHorizon stores the length of the i-th row in the current solution
      std::vector<int> mLeftHorizon;

      //the bottomHorizon stores the height of the i-th column in the current solution
      std::vector<int> mBottomHorizon;

      // inner horizons base and extent (number of free cells)
      std::vector<int> mInnerBottomHorizon;
      std::vector<int> mInnerBottomExtent;

      std::vector<std::vector<InnerGap>> bottomInnerGaps;
      std::vector<std::vector<InnerGap>> leftInnerGaps;

      std::vector<int> mInnerLeftHorizon;
      std::vector<int> mInnerLeftExtent;

      // Skyline event points (indices where the horizon changes)
      std::vector<uint64_t> mBottomEventBits;
      std::vector<uint64_t> mLeftEventBits;

      //the size of the packing grid
      vcg::Point2i mSize;

      //packing parameters
      Parameters params;

  public:
      packingfield(vcg::Point2i size, const Parameters& par)
      {
          mBottomHorizon.resize(size.X(), 0);
          mLeftHorizon.resize(size.Y(), 0);

          mInnerBottomHorizon.resize(size.X(), 0);
          mInnerBottomExtent.resize(size.X(), 0);

          bottomInnerGaps.resize(size.X(), {});
          leftInnerGaps.resize(size.X(), {});

          mInnerLeftHorizon.resize(size.Y(), 0);
          mInnerLeftExtent.resize(size.Y(), 0);

          params = par;
          mSize = Point2i(size.X(), size.Y());

          mBottomEventBits.assign((size.X() + 63) / 64, 0);
          mLeftEventBits.assign((size.Y() + 63) / 64, 0);

          setEvent(mBottomEventBits, 0);
          setEvent(mLeftEventBits, 0);
      }

      inline void setEvent(std::vector<uint64_t>& bits, int x) {
          if (x >= 0 && (size_t)(x / 64) < bits.size())
              bits[x / 64] |= (1ULL << (x % 64));
      }

      inline void clearEvent(std::vector<uint64_t>& bits, int x) {
          if (x >= 0 && (size_t)(x / 64) < bits.size())
              bits[x / 64] &= ~(1ULL << (x % 64));
      }

      template<typename Lambda>
      void forEachBottomEvent(Lambda callback) const {
          for (size_t i = 0; i < mBottomEventBits.size(); ++i) {
              uint64_t word = mBottomEventBits[i];
              while (word) {
                  int bit = vcg_ctzll(word);
                  callback(int(i * 64 + bit));
                  word &= ~(1ULL << bit);
              }
          }
      }

      template<typename Lambda>
      void forEachLeftEvent(Lambda callback) const {
          for (size_t i = 0; i < mLeftEventBits.size(); ++i) {
              uint64_t word = mLeftEventBits[i];
              while (word) {
                  int bit = vcg_ctzll(word);
                  callback(int(i * 64 + bit));
                  word &= ~(1ULL << bit);
              }
          }
      }

      std::vector<int>& bottomHorizon() { return mBottomHorizon; }
      std::vector<int>& leftHorizon() { return mLeftHorizon; }
      vcg::Point2i& size() { return mSize; }

      //returns the score relative to the left horizon of that poly in that particular position, taking into account the choosen algo
      int getCostX(RasterizedOutline2& poly, Point2i pos, int rast_i) {
          switch (params.costFunction) {
          case CostFuncEnum::MinWastedSpace: return emptyCellBetweenPolyAndLeftHorizon(poly, pos, rast_i);
          case CostFuncEnum::LowestHorizon: return maxXofPoly(poly, pos, rast_i);
          case CostFuncEnum::MixedCost: return costXWithPenaltyOnY(poly, pos, rast_i);
          }
          return 0;
      }

      //returns the score relative to the bottom horizon of that poly in that particular position, taking into account the choosen algo
      int getCostY(RasterizedOutline2& poly, Point2i pos, int rast_i) {
          switch (params.costFunction) {
          case CostFuncEnum::MinWastedSpace: return emptyCellBetweenPolyAndBottomHorizon(poly, pos, rast_i);
          case CostFuncEnum::LowestHorizon: return maxYofPoly(poly, pos, rast_i);
          case CostFuncEnum::MixedCost: return costYWithPenaltyOnX(poly, pos, rast_i);
          }
          return 0;
      }

      //given a poly and the column at which it is placed,
      //this returns the Y at which the wasted space is minimum
      //i.e. the Y at which the polygon touches the horizon
      int dropY(RasterizedOutline2& poly, int col, int rast_i) {
          std::vector<int>& bottom = poly.getBottom(rast_i);
          int w = bottom.size();
          if (w == 0) return 0;

          // SIMD optimization
          Eigen::Map<const Eigen::ArrayXi> bottomVec(bottom.data(), w);
          Eigen::Map<const Eigen::ArrayXi> horizonVec(&mBottomHorizon[col], w);
          int y_max = (horizonVec - bottomVec).maxCoeff();

          if (y_max + poly.gridHeight(rast_i) >= mSize.Y())
              return INVALID_POSITION;
          return y_max;
      }

      int dropYInner(RasterizedOutline2& poly, int col, int rast_i) {
          std::vector<int>& bottom = poly.getBottom(rast_i);
          std::vector<int>& deltaY = poly.getDeltaY(rast_i);
          int w = bottom.size();
          if (w == 0) return 0;

          // SIMD optimization
          Eigen::Map<const Eigen::ArrayXi> bottomVec(bottom.data(), w);
          Eigen::Map<const Eigen::ArrayXi> innerHorizonVec(&mInnerBottomHorizon[col], w);
          int y_max = (innerHorizonVec - bottomVec).maxCoeff();

          if (y_max + poly.gridHeight(rast_i) >= mSize.Y()) {
              return INVALID_POSITION;
          }
          // check if the placement is feasible
          for (int i = 0; i < w; ++i) {
              if (y_max + bottom[i] < mBottomHorizon[col + i]
                      && y_max + bottom[i] + deltaY[i] > mInnerBottomHorizon[col + i] + mInnerBottomExtent[col + i]) {
                  return INVALID_POSITION;
              }
          }
          return y_max;
      }

      //given a poly and the row at which it is placed,
      //this returns the X at which the wasted space is minimum
      //i.e. the X at which the polygon touches the left horizon
      int dropX(RasterizedOutline2& poly, int row, int rast_i) {
          std::vector<int>& left = poly.getLeft(rast_i);
          int h = left.size();
          if (h == 0) return 0;

          // SIMD optimization
          Eigen::Map<const Eigen::ArrayXi> leftVec(left.data(), h);
          Eigen::Map<const Eigen::ArrayXi> horizonVec(&mLeftHorizon[row], h);
          int x_max = (horizonVec - leftVec).maxCoeff();

          if (x_max + poly.gridWidth(rast_i) >= mSize.X())
              return INVALID_POSITION;
          return x_max;
      }

      int dropXInner(RasterizedOutline2& poly, int row, int rast_i) {
          std::vector<int> left = poly.getLeft(rast_i);
          std::vector<int> deltaX = poly.getDeltaX(rast_i);
          int h = left.size();
          if (h == 0) return 0;

          // SIMD optimization
          Eigen::Map<const Eigen::ArrayXi> leftVec(left.data(), h);
          Eigen::Map<const Eigen::ArrayXi> innerHorizonVec(&mInnerLeftHorizon[row], h);
          int x_max = (innerHorizonVec - leftVec).maxCoeff();

          if (x_max + poly.gridWidth(rast_i) >= mSize.X())
              return INVALID_POSITION;
          // sanity check
          for (int i = 0; i < h; ++i) {
              if (x_max + left[i] < mLeftHorizon[row + i]
                      && x_max + left[i] + deltaX[i] > mInnerLeftHorizon[row + i] + mInnerLeftExtent[row + i])
                  return INVALID_POSITION;
          }
          return x_max;
      }

      int costYWithPenaltyOnX(RasterizedOutline2& poly, Point2i pos, int rast_i) {
          std::vector<int>& left = poly.getLeft(rast_i);
          std::vector<int>& deltaX = poly.getDeltaX(rast_i);

          //get the standard cost on X axis
          int score = emptyCellBetweenPolyAndBottomHorizon(poly, pos, rast_i);

          //apply a penalty if the poly is the poly is far from the left horizon
          //thus preferring poly which are closer to the left horizon
          for (size_t i = 0; i < left.size(); ++i) {
              //ASSUMPTION: if the poly is (partially/fully) under the left horizon,
              //then we will count this as a good thing (subtracting a quantity from the cost) but since we don't have
              //a grid holding the current state of the packing field, we don't know the position of the polygons at our left side,
              //so we ASSUME that there isn't any polygon between the poly we're considering and the Y axis of the packing field,
              //and count the number of cells between us and the RIGHT end the packing field
              //(NOTE: ^^^^^^^ this implies that the closer we are to the left horizon, the lower the cost will get)
              if (pos.X() + left[i] < mLeftHorizon[pos.Y() + i])
                  //number of cells between us and the RIGHT end the packing field
                  score -= mSize.X() - pos.X() - left[i];
                  //score -= (pos.X() + left[i] + deltaX[i]);
              else         //the number of cells between the bottom side of the poly at the (posY+i)-th row and the value of the horizon in that row
                  score += pos.X() + left[i] - mLeftHorizon[pos.Y() + i];
          }
          return score;
      }

      /* Returns the number of empty cells between the poly's bottom side and the
       * bottom horizon. If the poly is below the bottom horizon, it returns the
       * distance between the poly's bottom and the grid bottom inverted in sign,
       * therefore leaving more space to possibly fit other polygons. */
      int emptyCellBetweenPolyAndBottomHorizon(RasterizedOutline2& poly, Point2i pos, int rast_i)
      {
          std::vector<int>& bottom = poly.getBottom(rast_i);
          int score = 0;
          for (size_t i = 0; i < bottom.size(); ++i) {
              if (pos.Y() + bottom[i] < mBottomHorizon[pos.X() + i])
                  score -=  pos.Y() + bottom[i];
              else
                  //count the number of empty cells between poly's bottom side and the bottom horizon
                  score +=  pos.Y() + bottom[i] - mBottomHorizon[pos.X() + i];
          }
          return score;
      }


      int costXWithPenaltyOnY(RasterizedOutline2& poly, Point2i pos, int rast_i) {
          std::vector<int>& bottom = poly.getBottom(rast_i);
          std::vector<int>& deltaY = poly.getDeltaY(rast_i);

          //get the standard cost on X axis
          int score = emptyCellBetweenPolyAndLeftHorizon(poly, pos, rast_i);

          //apply a penalty if the poly is the poly is far from the bottom horizon
          //thus preferring poly which are closer to the bottom horizon
          for (size_t i = 0; i < bottom.size(); ++i) {
              //ASSUMPTION: if the poly is (partially/fully) under the bottom horizon,
              //then we will count this as a good thing (subtracting a quantity from the cost) but since we don't have
              //a grid holding the current state of the packing field, we don't know the position of the polygons beneath us,
              //so we ASSUME that there isn't any polygon between the poly we're considering and the X axis of the packing field,
              //and count the number of cells between us and the TOP end the packing field
              //(NOTE: ^^^^^^^ this implies that the closer we are to the bottom horizon, the lower the cost will get)
              if (pos.Y() + bottom[i] < mBottomHorizon[pos.X() + i])
                  //number of cells between us and the TOP side the packing field
                  score -= (mSize.Y() - pos.Y() - bottom[i]);
                  //score -= (pos.Y() + bottom[i] + deltaY[i]);
              else         //the number of cells between the left side of the poly at the (posX+i)-th column and the value of the horizon in that column
                  score += pos.X() + bottom[i] - mBottomHorizon[pos.X() + i];
          }
          return score;
      }

      int maxYofPoly(RasterizedOutline2& poly, Point2i pos, int rast_i)
      {
          //return pos.Y() + poly.gridHeight(rast_i);

          int maxY = -INT_MAX;
          std::vector<int>& bottom = poly.getBottom(rast_i);
          std::vector<int>& deltaY = poly.getDeltaY(rast_i);
          for (unsigned i = 0; i < bottom.size(); ++i) {
              int yi = 0;
              if (pos.Y() + bottom[i] + deltaY[i] < mBottomHorizon[pos.X() + i]) {
                  yi = -(pos.Y() + bottom[i]);
              } else {
                  yi = pos.Y() + bottom[i] + deltaY[i];
              }
              if (yi > maxY)
                  maxY = yi;
          }
          return maxY;

      }

      int maxXofPoly(RasterizedOutline2& poly, Point2i pos, int rast_i)
      {
          //return pos.X() + poly.gridWidth(rast_i);

          int maxX = -INT_MAX;
          std::vector<int>& left = poly.getLeft(rast_i);
          std::vector<int>& deltaX = poly.getDeltaX(rast_i);
          for (unsigned i = 0; i < left.size(); ++i) {
              int xi = 0;
              if (pos.X() + left[i] + deltaX[i] < mLeftHorizon[pos.Y() + i]) {
                  xi = -(pos.X() + left[i]);
              } else {
                  xi = pos.X() + left[i] + deltaX[i];
              }
              if (xi > maxX)
                  maxX = xi;
          }
          return maxX;
      }

      /* Returns the number of empty cells between the poly's left side and the
       * left horizon. If the poly is below the left horizon, it returns the
       * distance between the poly's and grid left side inverted in sign. */
      int emptyCellBetweenPolyAndLeftHorizon(RasterizedOutline2& poly, Point2i pos, int rast_i)
      {
          std::vector<int>& left = poly.getLeft(rast_i);
          int score = 0;
          //count the number of empty cells between poly's left side and the left horizon
          for (size_t i = 0; i < left.size(); ++i) {
              if (pos.X() + left[i] < mLeftHorizon[pos.Y() + i])
                  score -= pos.X() + left[i];
              else
                  score += pos.X() + left[i] - mLeftHorizon[pos.Y() + i];
          }
          return score;
      }

      void updateEvents(std::vector<uint64_t>& bits, const std::vector<int>& horizon, int i, int max_i) {
          if (i <= 0 || i >= max_i) return;
          bool isEvent = (horizon[i] != horizon[i-1]);
          if (isEvent) {
              setEvent(bits, i);
          } else {
              clearEvent(bits, i);
          }
      }

      //updates the horizons according to the chosen position
      void placePoly(RasterizedOutline2& poly, Point2i pos, int rast_i) {

          std::vector<int>& bottom = poly.getBottom(rast_i);
          std::vector<int>& deltaY = poly.getDeltaY(rast_i);
          std::vector<int>& left = poly.getLeft(rast_i);
          std::vector<int>& deltaX = poly.getDeltaX(rast_i);

          //update bottom horizon
          for (int i = 0; i < poly.gridWidth(rast_i); i++) {
              int tmpHor = pos.Y() + bottom[i] + deltaY[i];
              std::vector<InnerGap>& bottomIgVec = bottomInnerGaps[pos.X() + i];

              if (tmpHor > mBottomHorizon[pos.X() + i]) {
                  // check if we create a bigger gap than the one currently tracked
                  // as the inner horizon. If we do, the current bottom horizon
                  // becomes the new inner horizon
                  int gapExtent = pos.Y() + bottom[i] - mBottomHorizon[pos.X() + i];
                  if (gapExtent < 0) {
                      // This can happen if the poly was placed using the left horizon
                      // and ends up filling both the inner and outer space at the same time
                      // just update the inner horizon extent...
                      if (mInnerBottomHorizon[pos.X() + i] < pos.Y() + bottom[i]
                              && mInnerBottomHorizon[pos.X() + i] + mInnerBottomExtent[pos.X() + i] > pos.Y() + bottom[i])
                          mInnerBottomExtent[pos.X() + i] = pos.Y() + bottom[i] - mInnerBottomHorizon[pos.X() + i];
                  }
                  //else if (gapExtent > mInnerBottomExtent[pos.X() + i]) {
                  //    mInnerBottomHorizon[pos.X() + i] = mBottomHorizon[pos.X() + i];
                  //    mInnerBottomExtent[pos.X() + i] = gapExtent;
                  //}
                  else {
                      // record the new gap
                      InnerGap ig{mBottomHorizon[pos.X() + i], gapExtent};
                      bottomIgVec.push_back(ig);
                      std::push_heap(bottomIgVec.begin(), bottomIgVec.end());

                      // also record the current gap
                      if (mInnerBottomExtent[pos.X() + i] > 0) {
                          ig.pos = mInnerBottomHorizon[pos.X() + i];
                          ig.extent = mInnerBottomExtent[pos.X() + i];
                          bottomIgVec.push_back(ig);
                          std::push_heap(bottomIgVec.begin(), bottomIgVec.end());
                      }
                  }
                  // then update the bottom horizon
                  mBottomHorizon[pos.X() + i] = tmpHor;
              } else {
                  // if the poly fills the space between the currently tracked
                  // inner bottom horizon and its extent, update the gap.
                  // Note that this update is local, since we only track the inner horizon and
                  // its extent. If bigger gaps exist, we lose track of them.
                  int bottomExtent = pos.Y() + bottom[i] - mInnerBottomHorizon[pos.X() + i];
                  int topExtent = mInnerBottomHorizon[pos.X() + i] + mInnerBottomExtent[pos.X() + i] - tmpHor;
                  if (bottomExtent >= 0 && topExtent >= 0) {
                      if (bottomExtent > topExtent) {
                          //mInnerBottomExtent[pos.X() + i] = bottomExtent;

                          InnerGap ig{mInnerBottomHorizon[pos.X() + i], bottomExtent};
                          bottomIgVec.push_back(ig);
                          std::push_heap(bottomIgVec.begin(), bottomIgVec.end());
                      } else {
                          //mInnerBottomHorizon[pos.X() + i] = tmpHor;
                          //mInnerBottomExtent[pos.X() + i] = topExtent;

                          InnerGap ig{tmpHor, topExtent};
                          bottomIgVec.push_back(ig);
                          std::push_heap(bottomIgVec.begin(), bottomIgVec.end());
                      }
                  } else {
                      // this is a tricky situation where the poly partially intersects the inner horizon
                      // TODO: properly update the extents, for now I just clear the inner horizon
                      mInnerBottomHorizon[pos.X() + i] = 0;
                      mInnerBottomExtent[pos.X() + i] = 0;
                  }
              }

              if (bottomIgVec.size() == 0) {
                  mInnerBottomHorizon[pos.X() + i] = 0;
                  mInnerBottomExtent[pos.X() + i] = 0;
              } else {
                  std::pop_heap(bottomIgVec.begin(), bottomIgVec.end());
                  InnerGap ig = bottomIgVec.back();
                  bottomIgVec.pop_back();
                  mInnerBottomHorizon[pos.X() + i] = ig.pos;
                  mInnerBottomExtent[pos.X() + i] = ig.extent;
              }
          }

          int x_start_b = pos.X();
          int w = poly.gridWidth(rast_i);
          for (int x = x_start_b; x <= x_start_b + w; ++x) {
              updateEvents(mBottomEventBits, mBottomHorizon, x, mSize.X());
          }

          //update left horizon
          for (int i = 0; i < poly.gridHeight(rast_i); i++) {
              int tmpHor = pos.X() + left[i] + deltaX[i];
              std::vector<InnerGap>& leftIgVec = leftInnerGaps[pos.Y() + i];

              if (tmpHor > mLeftHorizon[pos.Y() + i]) {
                  int gapExtent = pos.X() + left[i] - mLeftHorizon[pos.Y() + i];
                  if (gapExtent < 0) {
                      assert(0);
                      if (mInnerLeftHorizon[pos.Y() + i] < pos.X() + left[i]
                              && mInnerLeftHorizon[pos.Y() + i] + mInnerLeftExtent[pos.Y() + i] > pos.X() + left[i])
                          mInnerLeftExtent[pos.Y() + i] = pos.X() + left[i] - mInnerLeftHorizon[pos.Y() + i];
                  }
                  //else if (gapExtent > mInnerLeftExtent[pos.Y() + i]) {
                  //    mInnerLeftHorizon[pos.Y() + i] = mLeftHorizon[pos.Y() + i];
                  //    mInnerLeftExtent[pos.Y() + i] = gapExtent;
                  //}
                  else {
                      InnerGap ig{mLeftHorizon[pos.Y() + i], gapExtent};
                      leftIgVec.push_back(ig);
                      std::push_heap(leftIgVec.begin(), leftIgVec.end());
                      if (mInnerLeftExtent[pos.Y() + i] > 0) {
                          ig.pos = mInnerBottomHorizon[pos.Y() + i];
                          ig.extent = mInnerLeftExtent[pos.Y() + i];
                          leftIgVec.push_back(ig);
                          std::push_heap(leftIgVec.begin(), leftIgVec.end());
                      }
                  }
                  mLeftHorizon[pos.Y() + i] = tmpHor;
              } else {
                  int leftExtent = pos.X() + left[i] - mInnerLeftHorizon[pos.Y() + i];
                  int rightExtent = mInnerLeftHorizon[pos.Y() + i] + mInnerLeftExtent[pos.Y() + i] - tmpHor;
                  if (leftExtent >= 0 && rightExtent >= 0) {
                      if (leftExtent > rightExtent) {
                          //mInnerLeftExtent[pos.Y() + i] = leftExtent;

                          InnerGap ig{mInnerLeftHorizon[pos.Y() + i], leftExtent};
                          leftIgVec.push_back(ig);
                          std::push_heap(leftIgVec.begin(), leftIgVec.end());
                      } else {
                          //mInnerLeftHorizon[pos.Y() + i] = tmpHor;
                          //mInnerLeftExtent[pos.Y() + i] = rightExtent;
                          InnerGap ig{tmpHor, rightExtent};
                          leftIgVec.push_back(ig);
                          std::push_heap(leftIgVec.begin(), leftIgVec.end());
                      }
                  } else {
                      // this is a tricky situation where the poly partially intersects the inner horizon
                      // TODO: properly update the extents, for now I just clear the inner horizon
                      mInnerLeftHorizon[pos.Y() + i] = 0;
                      mInnerLeftExtent[pos.Y() + i] = 0;
                  }
              }

              if (leftIgVec.size() == 0) {
                  mInnerLeftHorizon[pos.Y() + i] = 0;
                  mInnerLeftExtent[pos.Y() + i] = 0;
              } else {
                  std::pop_heap(leftIgVec.begin(), leftIgVec.end());
                  InnerGap ig = leftIgVec.back();
                  leftIgVec.pop_back();
                  mInnerLeftHorizon[pos.Y() + i] = ig.pos;
                  mInnerLeftExtent[pos.Y() + i] = ig.extent;
              }

          }

          int y_start_l = pos.Y();
          int h = poly.gridHeight(rast_i);
          for (int y = y_start_l; y <= y_start_l + h; ++y) {
              updateEvents(mLeftEventBits, mLeftHorizon, y, mSize.Y());
          }
      }
  };


    static bool Pack(std::vector< std::vector< Point2x>  > &polyPointsVec,
                     Point2i containerSize,
                     std::vector<Similarity2x> &trVec,
                     const Parameters &packingPar)
    {
        std::vector<Point2i> containerSizes(1, containerSize);
        std::vector<int> polyToContainer;
        return Pack(polyPointsVec, containerSizes, trVec, polyToContainer, packingPar);
    }

    static bool Pack(std::vector<std::vector<Point2x>> &polyPointsVec,
                     const std::vector<Point2i> &containerSizes,
                     std::vector<Similarity2x> &trVec,
                     std::vector<int> &polyToContainer,
                     const Parameters &packingPar)
    {
        int containerNum = containerSizes.size();

        float gridArea = 0;
        //if containerSize isn't multiple of cell size, crop the grid (leaving containerSize as it is)
        for (int i = 0; i < containerNum; i++) {
            Point2i gridSize(containerSizes[i].X(),
                             containerSizes[i].Y());

            gridArea += (gridSize.X() * gridSize.Y());
        }

        float totalArea = 0;
        for (size_t j = 0; j < polyPointsVec.size(); j++) {
            float curArea = tri::OutlineUtil<SCALAR_TYPE>::Outline2Area(polyPointsVec[j]);
            if(curArea<0) tri::OutlineUtil<SCALAR_TYPE>::ReverseOutline2(polyPointsVec[j]);
            totalArea += fabs(curArea);
        }

        //we first set it to the "optimal" scale
        float optimalScale = sqrt(gridArea / totalArea);



        //create the vector of polys, starting for the poly points we received as parameter
        std::vector<RasterizedOutline2> polyVec(polyPointsVec.size());
        for(size_t i=0;i<polyVec.size();i++) {
            polyVec[i].setPoints(polyPointsVec[i]);
        }

        std::vector<std::vector<int>> trials = InitializePermutationVectors(polyPointsVec, packingPar);

        double bestEfficiency = 0;
        for (std::size_t i = 0; i < trials.size(); ++i) {

            float currScale = optimalScale;
            float latestFailScale = 0;

            std::vector<Similarity2x> trVecIter;
            std::vector<int> polyToContainerIter;

            bool ret = false;
            //we look for the first scale factor which makes the packing algo succeed
            //we will use this value in the bisection method afterwards
            ret = PolyPacking(polyPointsVec, containerSizes, trVecIter, polyToContainerIter, packingPar, currScale, polyVec, trials[i]);
            while (!ret) {
                //printf("Initial packing failed %d\n", k++);
                latestFailScale = currScale;
                currScale *= 0.60;
                ret = PolyPacking(polyPointsVec, containerSizes, trVecIter, polyToContainerIter, packingPar, currScale, polyVec, trials[i]);
            }

            //if it managed to pack with the optimal scale (VERY unlikely), skip bisection
            float latestSuccessScale = currScale;
            //int cnt = 1;
            assert(currScale <= optimalScale);
            if (currScale < optimalScale) {
                //BISECTION METHOD
                float tmpScale = (latestSuccessScale + latestFailScale) / 2;
                while ( (latestFailScale / latestSuccessScale) - 1 > 0.001
                        || ((latestFailScale / latestSuccessScale) - 1 < 0.001 && !ret) ) {

                    tmpScale = (latestSuccessScale + latestFailScale) / 2;
                    ret = PolyPacking(polyPointsVec, containerSizes, trVecIter, polyToContainerIter, packingPar, tmpScale, polyVec, trials[i]);
                    if (ret) latestSuccessScale = tmpScale;
                    else latestFailScale = tmpScale;
                    //cnt++;
                }
            }

            float finalArea = 0;
            //compute occupied area
            for (size_t j = 0; j < polyPointsVec.size(); j++) {
                std::vector<Point2f> oldPoints = polyPointsVec[j];
                for (size_t k = 0; k < oldPoints.size(); k++) {
                    oldPoints[k].Scale(latestSuccessScale, latestSuccessScale);
                }
                finalArea +=  tri::OutlineUtil<SCALAR_TYPE>::Outline2Area(oldPoints);
            }

            //printf("PACKING EFFICIENCY: %f with scale %f after %d attempts\n", finalArea/gridArea, latestSuccessScale, cnt);

            double efficiency = finalArea / gridArea;
            if (efficiency > bestEfficiency) {
                trVec = trVecIter;
                polyToContainer = polyToContainerIter;
                bestEfficiency = efficiency;
            }

        }

        return true;
    }

    static std::vector<std::vector<int>>
    InitializePermutationVectors(const std::vector<std::vector<Point2x>>& polyPointsVec,
                                 const Parameters& packingPar)
    {
        std::vector<std::vector<int>> trials;

        // Build a permutation that holds the indexes of the polys ordered by their area
        std::vector<int> perm(polyPointsVec.size());
        for(size_t i = 0; i < polyPointsVec.size(); i++)
            perm[i] = i;
        sort(perm.begin(), perm.end(), ComparisonFunctor<float>(polyPointsVec));

        trials.push_back(perm);

        // if packing with random permutations, compute a small number of randomized
        // sequences. Each random sequence is generated from the initial permutation
        // by shuffling only the larger polygons
        if (packingPar.permutations) {
            int minObjNum = std::min(5, int(perm.size()));
            float largestArea = tri::OutlineUtil<SCALAR_TYPE>::Outline2Area(polyPointsVec[perm[0]]);
            float thresholdArea = largestArea * 0.5;
            std::size_t i;
            for (i = 0; i < polyPointsVec.size(); ++i)
                if (tri::OutlineUtil<SCALAR_TYPE>::Outline2Area(polyPointsVec[perm[i]]) < thresholdArea)
                    break;
            int numPermutedObjects = std::max(minObjNum, int(i));
            int permutationCount = numPermutedObjects * 5;
            //printf("PACKING: trying %d random permutations of the largest %d elements\n", permutationCount, numPermutedObjects);
            for (int k = 0; k < permutationCount; ++k) {
                std::random_shuffle(perm.begin(), perm.begin() + numPermutedObjects);
                trials.push_back(perm);
            }
        }

        return trials;
    }

    static bool PackAtFixedScale(std::vector<std::vector<Point2x>> &polyPointsVec,
                     const std::vector<Point2i> &containerSizes,
                     std::vector<Similarity2x> &trVec,
                     std::vector<int> &polyToContainer,
                     const Parameters &packingPar,
                     float scale)
    {
        //create the vector of polys, starting for the poly points we received as parameter
        std::vector<RasterizedOutline2> polyVec(polyPointsVec.size());
        for(size_t i=0;i<polyVec.size();i++) {
            polyVec[i].setPoints(polyPointsVec[i]);
        }

        std::vector<std::vector<int>> trials = InitializePermutationVectors(polyPointsVec, packingPar);

        for (std::size_t i = 0; i < trials.size(); ++i) {
            std::vector<Similarity2x> trVecIter;
            std::vector<int> polyToContainerIter;
            if (PolyPacking(polyPointsVec, containerSizes, trVecIter, polyToContainerIter, packingPar, scale, polyVec, trials[i], false)) {
                trVec = trVecIter;
                polyToContainer = polyToContainerIter;
                return true;
            }
        }

        return false;
    }

    /*
     * Pack charts using a best effort policy. The idea is that this function
     * packs what it can in the given space without scaling the outlines.
     *
     * Returns the number of charts actually packed.
     *
     * Function parameters:
     *   outline2Vec (IN) vector of outlines to pack
     *   containerSizes (IN) vector of container (grid) sizes
     *   trVec (OUT) vector of transformations that must be applied to the objects
     *   polyToContainer (OUT) vector of outline-to-container mappings. If polyToContainer[i] == -1
     *     then outline i did not fit in the packing grids, and the transformation trVec[i] is meaningless
     * */
    static int
    PackBestEffort(std::vector<std::vector<Point2x>> &outline2Vec,
                   const std::vector<Point2i> &containerSizes,
                   std::vector<Similarity2x> &trVec,
                   std::vector<int> &polyToContainer,
                   const Parameters &packingPar)
    {
        return PackBestEffortAtScale(outline2Vec, containerSizes, trVec, polyToContainer, packingPar, 1.0f);
    }

    /* Same as PackBestEffort() but allows to specify the outlines scaling factor */
    static int
    PackBestEffortAtScale(std::vector<std::vector<Point2x>> &outline2Vec,
                          const std::vector<Point2i> &containerSizes,
                          std::vector<Similarity2x> &trVec,
                          std::vector<int> &polyToContainer,
                          const Parameters &packingPar, float scaleFactor)
    {
        std::vector<RasterizedOutline2> polyVec(outline2Vec.size());
        for(size_t i=0;i<polyVec.size();i++) {
            polyVec[i].setPoints(outline2Vec[i]);
        }

        polyToContainer.resize(outline2Vec.size(), -1);

        std::vector<std::vector<int>> trials = InitializePermutationVectors(outline2Vec, packingPar);
        int bestNumPlaced = 0;
        double bestPackedArea = 0;
        for (std::size_t i = 0; i < trials.size(); ++i) {
            std::vector<Similarity2x> trVecIter;
            std::vector<int> polyToContainerIter;
            PolyPacking(outline2Vec, containerSizes, trVecIter, polyToContainerIter, packingPar, scaleFactor, polyVec, trials[i], true);
            int numPlaced = 0;
            double packedArea = 0;
            for (std::size_t j = 0; j < polyToContainerIter.size(); ++j) {
                int k = polyToContainerIter[j];
                if (k != -1) {
                    packedArea += tri::OutlineUtil<SCALAR_TYPE>::Outline2Area(outline2Vec[j]);
                    numPlaced++;
                }
            }
            if (packedArea > bestPackedArea) {
                trVec = trVecIter;
                polyToContainer = polyToContainerIter;
                bestPackedArea = packedArea;
                bestNumPlaced = numPlaced;
            }
        }

        return bestNumPlaced;
    }

    //tries to pack polygons using the given gridSize and scaleFactor
    //stores the result, i.e. the vector of similarities, in trVec
    static bool PolyPacking(std::vector< std::vector< Point2x>  > &outline2Vec,
                            const std::vector<Point2i> &containerSizes,
                            std::vector<Similarity2x> &trVec,
                            std::vector<int> &polyToContainer,
                            const Parameters &packingPar,
                            float scaleFactor,
                            std::vector<RasterizedOutline2>& polyVec,
                            const std::vector<int>& perm,
                            bool bestEffort = false)
    {
        int containerNum = containerSizes.size();

        polyToContainer.clear();
        polyToContainer.resize(outline2Vec.size());
        trVec.resize(outline2Vec.size());

        //create packing fields, one for each container
        std::vector<Point2i> gridSizes;
        std::vector<packingfield> packingFields;
        for (int i=0; i < containerNum; i++) {
            gridSizes.push_back(Point2i(containerSizes[i].X(),
                                        containerSizes[i].Y()));

            packingfield one(gridSizes[i], packingPar);
            packingFields.push_back(one);
        }


        // **** Main Loop: Iterate sequentially over polys, but find best position in parallel ****
        for (size_t currPoly = 0; currPoly < polyVec.size(); currPoly++) {

            int i = perm[currPoly];

            // +++ Step 1: Just-In-Time Rasterization (Memory Safe) +++
            // Rasterize the multiple rotations for *only the current chart* in parallel.
            polyVec[i].resetState(packingPar.rotationNum);
            for (int rast_i = 0; rast_i < packingPar.rotationNum/4; rast_i++) {
                //create the rasterization (i.e. fills bottom/top/grids/internalWastedCells arrays)
                RASTERIZER_TYPE::rasterize(polyVec[i], scaleFactor, rast_i, packingPar.rotationNum, packingPar.gutterWidth);
            }

            // +++ Step 2: Parallel Placement Search +++
            PlacementResult bestOverallResult;

            #pragma omp parallel if(!omp_in_parallel())
            {
                PlacementResult bestThreadResult; // Each thread has its own local best result

                // Parallelize the search over all rotations. Use a dynamic schedule for better
                // load balancing, as work per rotation can be highly variable.
                #pragma omp for schedule(dynamic, 1)
                for (int rast_i = 0; rast_i < packingPar.rotationNum; rast_i++) {

                    //try to fit the poly in all containers, in all valid positions
                    for (int grid_i = 0; grid_i < containerNum; grid_i++) {
                        int maxCol = gridSizes[grid_i].X() - polyVec[i].gridWidth(rast_i);
                        int maxRow = gridSizes[grid_i].Y() - polyVec[i].gridHeight(rast_i);

                        //look for the best position, dropping from top
                        std::vector<uint8_t> candidateBits((maxCol + 8) / 8, 0);
                        auto setCandidate = [&](int col) {
                            if (col >= 0 && col <= maxCol) candidateBits[col >> 3] |= (1 << (col & 7));
                        };

                        setCandidate(0);
                        setCandidate(maxCol);

                        int poly_w = polyVec[i].gridWidth(rast_i);
                        packingFields[grid_i].forEachBottomEvent([&](int e) {
                            setCandidate(e);
                            setCandidate(e - 1);
                            setCandidate(e + 1);
                            setCandidate(e - poly_w);
                            setCandidate(e - poly_w + 1);
                            setCandidate(e - poly_w - 1);
                        });

                        std::vector<int> candidateCols;
                        candidateCols.reserve(128);
                        for (int col = 0; col <= maxCol; ++col) {
                            if (candidateBits[col >> 3] & (1 << (col & 7))) {
                                candidateCols.push_back(col);
                            }
                        }

                        for (int col : candidateCols) {
                            int currPolyY;
                            // Check primary horizon
                            currPolyY = packingFields[grid_i].dropY(polyVec[i],col, rast_i);
                            if (currPolyY != INVALID_POSITION) {
                                int currCost = packingFields[grid_i].getCostY(polyVec[i], Point2i(col, currPolyY), rast_i);
                                if (packingPar.doubleHorizon && (packingPar.minmax == true))
                                    currCost += packingFields[grid_i].getCostX(polyVec[i], Point2i(col, currPolyY), rast_i);
                                if (currCost < bestThreadResult.cost) {
                                    bestThreadResult.container = grid_i;
                                    bestThreadResult.cost = currCost;
                                    bestThreadResult.rastIndex = rast_i;
                                    bestThreadResult.polyX = col;
                                    bestThreadResult.polyY = currPolyY;
                                    bestThreadResult.placedUsingSecondaryHorizon = false;
                                }
                            }
                            // Check inner horizon
                            if (packingPar.innerHorizon) {
                                currPolyY = packingFields[grid_i].dropYInner(polyVec[i],col, rast_i);
                                if (currPolyY != INVALID_POSITION) {
                                    int currCost = packingFields[grid_i].getCostY(polyVec[i], Point2i(col, currPolyY), rast_i);
                                    if (packingPar.doubleHorizon && (packingPar.minmax == true))
                                        currCost += packingFields[grid_i].getCostX(polyVec[i], Point2i(col, currPolyY), rast_i);
                                    if (currCost < bestThreadResult.cost) {
                                        bestThreadResult.container = grid_i;
                                        bestThreadResult.cost = currCost;
                                        bestThreadResult.rastIndex = rast_i;
                                        bestThreadResult.polyX = col;
                                        bestThreadResult.polyY = currPolyY;
                                        bestThreadResult.placedUsingSecondaryHorizon = true;
                                    }
                                }
                            }
                        }

                        if (!packingPar.doubleHorizon)
                            continue;
                        
                        //look for the best position, dropping from left
                        std::vector<uint8_t> candidateBits((maxRow + 8) / 8, 0);
                        auto setCandidate = [&](int row) {
                            if (row >= 0 && row <= maxRow) candidateBits[row >> 3] |= (1 << (row & 7));
                        };

                        setCandidate(0);
                        setCandidate(maxRow);

                        int poly_h = polyVec[i].gridHeight(rast_i);
                        packingFields[grid_i].forEachLeftEvent([&](int e) {
                            setCandidate(e);
                            setCandidate(e - 1);
                            setCandidate(e + 1);
                            setCandidate(e - poly_h);
                            setCandidate(e - poly_h + 1);
                            setCandidate(e - poly_h - 1);
                        });

                        std::vector<int> candidateRows;
                        candidateRows.reserve(128);
                        for (int row = 0; row <= maxRow; ++row) {
                            if (candidateBits[row >> 3] & (1 << (row & 7))) {
                                candidateRows.push_back(row);
                            }
                        }

                        for (int row : candidateRows) {
                            int currPolyX;
                            // Check primary horizon
                            currPolyX = packingFields[grid_i].dropX(polyVec[i],row, rast_i);
                            if (currPolyX != INVALID_POSITION) {
                                int currCost = packingFields[grid_i].getCostX(polyVec[i], Point2i(currPolyX, row), rast_i);
                                if (packingPar.doubleHorizon && (packingPar.minmax == true))
                                    currCost += packingFields[grid_i].getCostY(polyVec[i], Point2i(currPolyX, row), rast_i);
                                if (currCost < bestThreadResult.cost) {
                                    bestThreadResult.container = grid_i;
                                    bestThreadResult.cost = currCost;
                                    bestThreadResult.rastIndex = rast_i;
                                    bestThreadResult.polyX = currPolyX;
                                    bestThreadResult.polyY = row;
                                    bestThreadResult.placedUsingSecondaryHorizon = false;
                                }
                            }
                            // Check inner horizon
                            if (packingPar.innerHorizon) {
                                currPolyX = packingFields[grid_i].dropXInner(polyVec[i],row, rast_i);
                                if (currPolyX != INVALID_POSITION) {
                                    int currCost = packingFields[grid_i].getCostX(polyVec[i], Point2i(currPolyX, row), rast_i);
                                    if (packingPar.doubleHorizon && (packingPar.minmax == true))
                                        currCost += packingFields[grid_i].getCostY(polyVec[i], Point2i(currPolyX, row), rast_i);
                                    if (currCost < bestThreadResult.cost) {
                                        bestThreadResult.container = grid_i;
                                        bestThreadResult.cost = currCost;
                                        bestThreadResult.rastIndex = rast_i;
                                        bestThreadResult.polyX = currPolyX;
                                        bestThreadResult.polyY = row;
                                        bestThreadResult.placedUsingSecondaryHorizon = true;
                                    }
                                }
                            }
                        }
                    }
                } // end of parallel for loop over rotations

                // Reduce Step: Each thread compares its local best against the shared global best.
                // The critical section ensures this comparison and update is thread-safe.
                #pragma omp critical
                {
                    if (bestThreadResult.cost < bestOverallResult.cost) {
                        bestOverallResult = bestThreadResult;
                    }
                }
            } // end of parallel block


            // +++ Step 3: Sequential State Update +++
            if (bestOverallResult.rastIndex == -1) {
                if (bestEffort) {
                    polyToContainer[i] = -1;
                    trVec[i] = {};
                } else {
                    return false;
                }
            } else {
                // Place the polygon, which updates the horizons in the 'packingFields' object.
                packingFields[bestOverallResult.container].placePoly(polyVec[i], Point2i(bestOverallResult.polyX, bestOverallResult.polyY), bestOverallResult.rastIndex);

                // Create the final similarity transform for this chart.
                float angleRad = float(bestOverallResult.rastIndex)*(M_PI*2.0)/float(packingPar.rotationNum);
                Box2f bb;
                std::vector<Point2f> points = polyVec[i].getPoints();
                for(size_t p_idx=0; p_idx<points.size(); ++p_idx) {
                    Point2f pp=points[p_idx];
                    pp.Rotate(angleRad);
                    bb.Add(pp);
                }

                polyToContainer[i] = bestOverallResult.container;

                float polyXInImgCoords = bestOverallResult.polyX;
                float scaledBBWidth = bb.DimX()*scaleFactor;
                float polyWidthInImgCoords = polyVec[i].gridWidth(bestOverallResult.rastIndex);
                float offsetX = (polyWidthInImgCoords - ceil(scaledBBWidth))/2.0;
                float scaledBBMinX = bb.min.X()*scaleFactor;

                float imgHeight = containerSizes[bestOverallResult.container].Y();
                float polyYInImgCoords = bestOverallResult.polyY;
                float polyHeightInImgCoords = polyVec[i].gridHeight(bestOverallResult.rastIndex);
                float topPolyYInImgCoords = polyYInImgCoords + polyHeightInImgCoords;
                float scaledBBHeight = bb.DimY()*scaleFactor;
                float offsetY = (polyHeightInImgCoords - ceil(scaledBBHeight))/2.0;
                float scaledBBMinY = bb.min.Y()*scaleFactor;
                trVec[i].tra = Point2f(polyXInImgCoords - scaledBBMinX + offsetX,
                                       imgHeight - topPolyYInImgCoords - scaledBBMinY + offsetY);
                trVec[i].rotRad = angleRad;
                trVec[i].sca = scaleFactor;
            }
        }

        return true;
    }

}; // end class



} // end namespace vcg

#endif // __RASTERIZED_OUTLINE2_PACKER_H__
