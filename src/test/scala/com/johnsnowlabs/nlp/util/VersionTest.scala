/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.Version
import org.junit.Assert.{assertFalse, assertTrue}
import org.scalatest.flatspec.AnyFlatSpec

class VersionTest extends AnyFlatSpec {

  "Version" should "cast to float version of 1 digit" taggedAs FastTest in {

    val actualVersion1 = Version(1).toFloat
    val actualVersion15 = Version(15).toFloat

    assert(actualVersion1 == 1f)
    assert(actualVersion15 == 15f)

  }

  it should "cast to float version of 2 digits" taggedAs FastTest in {
    val actualVersion1_2 = Version(List(1, 2)).toFloat
    val actualVersion2_7 = Version(List(2, 7)).toFloat

    assert(actualVersion1_2 == 1.2f)
    assert(actualVersion2_7 == 2.7f)
  }

  it should "cast to float version of 3 digits" taggedAs FastTest in {
    val actualVersion1_2_5 = Version(List(1, 2, 5)).toFloat
    val actualVersion3_2_0 = Version(List(3, 2, 0)).toFloat
    val actualVersion2_0_6 = Version(List(2, 0, 6)).toFloat

    assert(actualVersion1_2_5 == 1.25f)
    assert(actualVersion3_2_0 == 3.2f)
    assert(actualVersion2_0_6 == 2.06f)
  }

  it should "raise error when casting to float version > 3 digits" taggedAs FastTest in {
    assertThrows[UnsupportedOperationException] {
      Version(List(3, 0, 2, 5)).toFloat
    }
  }

  it should "be compatible for latest versions" taggedAs FastTest in {
    var currentVersion = Version(List(1, 2, 3))
    var modelVersion = Version(List(1, 2))

    var isCompatible = Version.isCompatible(currentVersion, modelVersion)

    assertTrue(isCompatible)

    currentVersion = Version(List(3, 0))
    modelVersion = Version(List(2, 4))

    isCompatible = Version.isCompatible(currentVersion, modelVersion)

    assertTrue(isCompatible)

    currentVersion = Version(List(2, 4, 5))
    modelVersion = Version(List(2, 4, 3))

    isCompatible = Version.isCompatible(currentVersion, modelVersion)

    assertTrue(isCompatible)

  }

  it should "be not compatible for latest versions" taggedAs FastTest in {
    var currentVersion = Version(List(1, 2))
    var modelVersion = Version(List(1, 2, 3))

    var isNotCompatible = Version.isCompatible(currentVersion, modelVersion)

    assertFalse(isNotCompatible)

    currentVersion = Version(List(2, 4))
    modelVersion = Version(List(3, 0))

    isNotCompatible = Version.isCompatible(currentVersion, modelVersion)

    assertFalse(isNotCompatible)

    currentVersion = Version(List(2, 4, 3))
    modelVersion = Version(List(2, 4, 5))

    isNotCompatible = Version.isCompatible(currentVersion, modelVersion)

    assertFalse(isNotCompatible)
  }

  it should "parse a version with fewer than 3 numbers" taggedAs FastTest in {
    val someVersion = "3.2"
    val expectedVersion = "3.2"
    val expectedFloatVersion = 3.2f
    val actualVersion = Version.parse(someVersion)

    assert(expectedVersion == actualVersion.toString)
    assert(expectedFloatVersion == actualVersion.toFloat)
  }

  it should "parse a version with 3 numbers" taggedAs FastTest in {
    val someVersion = "3.4.2"
    val expectedFloatVersion = 3.42f
    val actualVersion = Version.parse(someVersion)

    assert(someVersion == actualVersion.toString)
    assert(expectedFloatVersion == actualVersion.toFloat)
  }

  it should "truncate a version to 3 digits when it has more than 3 digits" taggedAs FastTest in {
    val someVersion = "3.5.1.5.4.20241007.4"
    val expectedVersion = "3.5.1"
    val expectedFloatVersion = 3.51f
    val actualVersion = Version.parse(someVersion)

    assert(expectedVersion == actualVersion.toString)
    assert(expectedFloatVersion == actualVersion.toFloat)
  }

  it should "handle a version with missing parts" taggedAs FastTest in {
    val someVersion = "3"
    val expectedVersion = "3"
    val expectedFloatVersion = 3.0f
    val actualVersion = Version.parse(someVersion)

    assert(expectedVersion == actualVersion.toString)
    assert(expectedFloatVersion == actualVersion.toFloat)
  }

  it should "handle a version with 3 digits and additional suffix" taggedAs FastTest in {
    val someVersion = "3.4.2-beta"
    val expectedVersion = "3.4.2"
    val expectedFloatVersion = 3.42f
    val actualVersion = Version.parse(someVersion)

    assert(expectedVersion == actualVersion.toString)
    assert(expectedFloatVersion == actualVersion.toFloat)
  }

  it should "raise exception with non-numeric and no valid parts" taggedAs FastTest in {
    val someVersion = "alpha.beta.gamma"

    assertThrows[UnsupportedOperationException] {
      Version.parse(someVersion).toFloat
    }
  }

  it should "handle a version with mixed numeric and non-numeric parts" taggedAs FastTest in {
    val someVersion = "3.4-alpha.2"
    val expectedVersion = "3.4"
    val expectedFloatVersion = 3.4f
    val actualVersion = Version.parse(someVersion)

    assert(expectedVersion == actualVersion.toString)
    assert(expectedFloatVersion == actualVersion.toFloat)
  }

}
